# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np
import pytest
from types import MappingProxyType
import json
from pathlib import Path
import shutil
import tempfile
from contextlib import ExitStack
from tvm.relay.op.contrib import cmsisnn

import tvm
from tvm import relay
import tvm.micro.testing
from tvm.relay.backend import Executor
from tvm.contrib import graph_executor, utils
from tvm import meta_schedule as ms
from tvm.micro.project_api import server

def get_low_high_atol_rtol(dtype):
    """Returns a tuple with boundary values and and tolerance for ACL tests."""

    if dtype == "float32":
        low, high, atol, rtol = (-127, 128, 0.001, 0.001)
    elif dtype == "uint8":
        low, high, atol, rtol = (0, 255, 1, 0)
    elif dtype == "int8":
        low, high, atol, rtol = (-127, 128, 1, 0)
    elif dtype == "int16":
        low, high, atol, rtol = (-127, 128, 1, 0)

    else:
        raise Exception(f"dtype not expected: {dtype}")

    return low, high, atol, rtol

def _get_qnn_params(input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w, channels):
    """Get output qnn parameters given input and kernel parameters."""
    input_max = input_sc * (255 - input_zp)
    input_min = -input_sc * input_zp
    kernel_max = kernel_sc * (255 - kernel_zp)
    kernel_min = -kernel_sc * kernel_zp
    output_limits = [
        kernel_max * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_min,
        kernel_max * kernel_h * kernel_w * channels * input_min,
    ]
    output_max = max(output_limits)
    output_min = min(output_limits)
    output_sc = (output_max - output_min) / 255
    output_zp = -int(output_min / output_sc)
    return output_zp, output_sc

def _get_qnn_model(
    shape,
    kernel_h,
    kernel_w,
    padding,
    strides,
    dilation,
    groups,
    dtype,
    channels,
    input_zp,
    input_sc,
    kernel_zp,
    kernel_sc,
    output_zp,
    output_sc,
    var_names,
    has_bias=False,
    has_activation=False,
    has_pad=False,
    use_cmsis_nn = False,
):
    """Return a model and any parameters it may have."""
    low, high, _, _ = get_low_high_atol_rtol(dtype)

    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    if has_pad:
        p = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0))
        a = relay.nn.pad(a, pad_width=p, pad_value=input_zp, pad_mode="constant")
        padding = (0, 0, 0, 0)
    else:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        shape = (shape[0], shape[1] + padding[0] * 2, shape[2] + padding[1] * 2, shape[3])
    is_depthwise = shape[3] == channels == groups
    
    if use_cmsis_nn:
        weight_format = "IOHW" if is_depthwise else "HWIO"
        if weight_format == "HWIO":
            weight_shape = (kernel_h, kernel_w, shape[3] // groups, channels)
        else:
            weight_shape = (shape[3] // groups, channels, kernel_h, kernel_w)
        bias_index = 3
        requantize_dtype = "int8"
    else:
        weight_format = "IOHW" if is_depthwise else "OHWI"
        if weight_format == "OHWI":
            weight_shape = (channels, kernel_h, kernel_w, shape[3] // groups)
        else:
            weight_shape = (shape[3] // groups, channels, kernel_h, kernel_w)
        bias_index = 0
        requantize_dtype = "int16"
     
    w = tvm.nd.array(np.random.uniform(low, high, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.qnn.op.conv2d(
        a,
        weights,
        input_zero_point=relay.const(input_zp, "int32"),
        kernel_zero_point=relay.const(kernel_zp, "int32"),
        input_scale=relay.const(input_sc, "float32"),
        kernel_scale=relay.const(kernel_sc, "float32"),
        kernel_size=(kernel_h, kernel_w),
        data_layout="NHWC",
        kernel_layout=weight_format,
        dilation=dilation,
        strides=strides,
        padding=padding,
        groups=groups,
        channels=channels,
        out_dtype="int32",
    )
    params = {"w": w}
    if has_bias:
        bias_shape = weight_shape[1] if is_depthwise else weight_shape[bias_index]
        b = tvm.nd.array(np.random.uniform(-128, 127, bias_shape).astype("int32"))
        # b = tvm.nd.array(np.zeros(bias_shape).astype("int32"))
        biasc = relay.const(b, "int32")
        out = relay.nn.bias_add(out, biasc, axis=3)
        params["b"] = b
    if has_activation:
        out = relay.nn.relu(out)
    req = relay.qnn.op.requantize(
        out,
        relay.const(input_sc * np.full((channels), kernel_sc), "float32"),  # input scale
        relay.const(0, "int32"),  # input zero point
        relay.const(output_sc, "float32"),  # output scale
        relay.const(output_zp, "int32"),  # output zero point
        axis=3,
        out_dtype=requantize_dtype,
    )
    return req, params

def create_relay_module(
    kernel_h,
    kernel_w,
    pad,
    stride,
    dilation,
    out_channels,
    shape,
    composite,
    is_depthwise,
    use_cmsis_nn,
):
    dtype = "int8"
    
    shape = (1, *shape)
    if is_depthwise:
        groups = shape[3]
    else:
        groups = 1
    outputs = []
    inputs = {"input": tvm.nd.array(np.random.uniform(0, 255, shape).astype(dtype))}

    input_zp = 5
    input_sc = 2.0
    kernel_zp = 0
    kernel_sc = 1.0
    output_zp, output_sc = _get_qnn_params(
        input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w, shape[3]
    )

    func, params = _get_qnn_model(
        shape,
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        groups,
        dtype,
        out_channels,
        input_zp,
        input_sc,
        kernel_zp,
        kernel_sc,
        output_zp,
        output_sc,
        iter(inputs),
        has_pad=composite[0],
        has_bias=composite[1],
        has_activation=composite[2],
        use_cmsis_nn=use_cmsis_nn,
    )

    mod = relay.Function(relay.analysis.free_vars(func), func)
    mod = tvm.IRModule.from_expr(mod)
    
    model_info = {
        "in_tensor": "input",
        "in_shape": shape,
        "in_dtype": dtype,
    }
    return mod, params, model_info

def create_aot_session(
    platform,
    board,
    target,
    mod,
    params,
    build_dir=Path(tempfile.mkdtemp()),
    tune_logs=None,
    timeout_override=None,
    use_cmsis_nn=False,
    project_options=None,
    use_existing=False,
):
    """AOT-compiles and uploads a model to a microcontroller, and returns the RPC session"""

    runtime = tvm.relay.backend.Runtime("crt", {"system-lib": True})

    USE_CMSIS_NN = use_cmsis_nn
    ENABLE_USMP = False
    USE_UPDATED_SCHEDULES = True

    if ENABLE_USMP:
        executor = tvm.relay.backend.Executor("aot", {"workspace-byte-alignment": 8})
    else:
        executor = tvm.relay.backend.Executor("aot")

    with ExitStack() as stack:
        
        disabled_pass = None
        config = {"tir.disable_vectorize": True}
        if ENABLE_USMP:
            config["tir.usmp.enable"]= True
        if USE_CMSIS_NN:
            config["relay.ext.cmsisnn.options"] = {"mcpu": target.mcpu}
        else:
            if (USE_UPDATED_SCHEDULES):
                config["relay.backend.use_meta_schedule"] = True
                config[ "relay.backend.tir_converter"] = "allow_extern"
                disabled_pass=["qnn.Legalize"]

        stack.enter_context(tvm.transform.PassContext(opt_level=3, config=config, disabled_pass=disabled_pass))
        if USE_CMSIS_NN:
            mod = cmsisnn.partition_for_cmsisnn(mod, params, mcpu=target.mcpu)
        elif USE_UPDATED_SCHEDULES:
            def schedule_fn(_sch):
                return True
            stack.enter_context(ms.database.ScheduleFnDatabase(schedule_fn))

        lowered = tvm.relay.build(
            mod,
            target=target,
            params=params,
            runtime=runtime,
            executor=executor,
            )
    parameter_size = len(tvm.runtime.save_param_dict(lowered.get_params()))
    print(f"Model parameter size: {parameter_size}")

    project_options = {
        "board": board,
        "project_type": "host_driven",
        # {} shouldn't be the default value for project options ({}
        # is mutable), so we use this workaround
        **(project_options or {}),
    }

    if use_existing:
        shutil.rmtree(build_dir / "project" / "build")
        project = tvm.micro.GeneratedProject.from_directory(
            build_dir / "project",
            options=project_options,
        )

    else:
        project = tvm.micro.generate_project(
            str(tvm.micro.get_microtvm_template_projects(platform)),
            lowered,
            build_dir / "project",
            project_options,
        )

    project.build()
    project.flash()
    return tvm.micro.Session(project.transport(), timeout_override=timeout_override)


def test_conv2d(
    platform, board, serial_number
):
    """Test a single conv2d"""
    use_cmsis_nn = False
    use_existing = True
    np.random.seed(0)
    trials = [
        # Normal convolution
        # [3, 3, (1, 1), (2, 2), (1, 1), 8, (96, 96, 3), (False, True, False), False],
        # [1, 1, (0,0), (1, 1), (1, 1), 16, (48, 48, 8), (False, True, False), False],
        # [1, 1, (0,0), (1, 1), (1, 1), 32, (24, 24, 16), (False, True, False), False],
        # [1, 1, (0,0), (1, 1), (1, 1), 32, (24, 24, 32), (False, True, False), False],
        # [1, 1, (0,0), (1, 1), (1, 1), 64, (12, 12, 32), (False, True, False), False],
        # [1, 1, (0,0), (1, 1), (1, 1), 64, (12, 12, 64), (False, True, False), False],
        # [1, 1, (0,0), (1, 1), (1, 1), 128, (6, 6, 64), (False, True, False), False],
        [1, 1, (0,0), (1, 1), (1, 1), 128, (6, 6, 128), (False, True, False), False],
        # [1, 1, (0,0), (1, 1), (1, 1), 256, (3, 3, 128), (False, True, False), False],
        # [1, 1, (0,0), (1, 1), (1, 1), 256, (3,3, 256), (False, True, False), False],
        # [1, 1, (0,0), (1, 1), (1, 1), 128, (1, 1, 640), (False, True, False), False],

        # Depth-wise convolution
        # [3, 3, (1, 1), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, True), True],
        # [5, 5, (2, 2), (1, 1), (1, 1), 20, (20, 20, 20), (False, True, False), True],
        # [3, 3, (2, 2), (2, 2), (1, 1), 14, (10, 10, 14), (True, False, False), True],
        # [5, 5, (0, 0), (1, 1), (1, 1), 20, (20, 20, 20), (False, False, False), True],
        # [3, 3, (1, 1), (2, 2), (1, 1), 14, (10, 10, 14), (False, True, True), True],
    ]
    results = []

    for (
        kernel_h,
        kernel_w,
        pad,
        stride,
        dilation,
        out_channels,
        shape,
        composite,
        is_depthwise,
    ) in trials:
        mod, params, model_info = create_relay_module(
            kernel_h,
            kernel_w,
            pad,
            stride,
            dilation,
            out_channels,
            shape,
            composite,
            is_depthwise,
            use_cmsis_nn,
        )
        print(mod["main"])
        input_name = model_info["in_tensor"]
        input_shape = model_info["in_shape"]
        input_dtype = model_info["in_dtype"]
        data_sample = tvm.nd.array(np.random.uniform(-128, 127, input_shape).astype(input_dtype))

        platform = "zephyr"
        options = {
                "config_main_stack_size": 4096,
                "serial_number": serial_number,
                "cmsis_path":"/home/mkatanbaf/cmsis/",
                }

        boards_file = Path(tvm.micro.get_microtvm_template_projects("zephyr")) / "boards.json"
        with open(boards_file) as f:
            boards = json.load(f)
        target = tvm.micro.testing.get_target("zephyr", board)

        if use_cmsis_nn:
            work_dir = Path("/home/mkatanbaf/microtvm/debug1/CMSIS")
        else:
            work_dir = Path("/home/mkatanbaf/microtvm/debug1/Native")
        if not use_existing:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir()

        with create_aot_session(
            platform,
            board,
            target,
            mod,
            params,
            #tune_logs=tune_logs,
            build_dir=work_dir,#tmpdir,
            timeout_override=server.TransportTimeouts(
                session_start_retry_timeout_sec=2,
                session_start_timeout_sec=20,
                session_established_timeout_sec=20,
            ),
            project_options=options,
            use_cmsis_nn = use_cmsis_nn,
            use_existing= use_existing,
        ) as session:
            aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
            aot_executor.get_input(0).copyfrom(data_sample)
            result = aot_executor.module.time_evaluator("run", session.device, number=3)()
            output = aot_executor.get_output(0).numpy()
        # print("time: ", result)
        results.append(result)
        # print(output)
        
        if not use_cmsis_nn:
            # Build reference model (without tuning)
            dev = tvm.cpu()
            target = tvm.micro.testing.get_target("crt")
            runtime = relay.backend.Runtime("crt", {"system-lib": True})

            # We assume our model's heavily-layout sensitive operators only consist of nn.conv2d
            desired_layouts = {'qnn.conv2d': ['NHWC', 'HWIO']}
            seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                            relay.transform.ConvertLayout(desired_layouts)])
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)

            with tvm.transform.PassContext(
                opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["AlterOpLayout"]
            ):
                ref_mod = relay.build(
                    mod,
                    target=target,
                    params=params,
                    runtime=runtime,
                )
            ref_mod.export_library(work_dir / "compiled_lib2.so")
            mod2: tvm.runtime.Module = tvm.runtime.load_module(work_dir / "compiled_lib2.so")
            graph_mod = graph_executor.GraphModule(mod2["default"](dev))
            graph_mod.set_input(input_name, data_sample)
            graph_mod.run()
            ref_output = graph_mod.get_output(0).numpy()
            
            print(output[0][0][0])
            print(ref_output[0][0][0])
            # assert np.allclose(output, ref_output, rtol=1e-4, atol=2e-4), "FAILED"
    for result in results:
        print(result)

