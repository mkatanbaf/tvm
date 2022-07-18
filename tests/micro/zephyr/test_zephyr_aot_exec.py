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
import logging
import os
import pathlib
import sys
import logging

import pytest
import numpy as np

import onnx
from PIL import Image

import tvm
import tvm.testing
import tvm.relay as relay
from tvm.relay.backend import Executor, Runtime
from tvm.relay.testing import byoc
from tvm.contrib import utils
from tvm.micro.testing.utils import check_tune_log
from tvm._ffi import get_global_func, register_func
from tvm.testing.aot import generate_ref_data

import test_utils


def _make_session(temp_dir, zephyr_board, west_cmd, mod, build_config, use_fvp):
    config_main_stack_size = None
    if test_utils.qemu_boards(zephyr_board):
        # fyi: qemu_riscv64 seems to be the greediest stack user
        config_main_stack_size = 4096
    else:
        # increase stack size for HW platforms
        config_main_stack_size = 2048

    project_options = {
        "project_type": "host_driven",
        "west_cmd": west_cmd,
        "verbose": bool(build_config.get("debug")),
        "zephyr_board": zephyr_board,
        "arm_fvp_path": "/opt/arm/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55",
        "use_fvp": bool(use_fvp),
    }
    if config_main_stack_size is not None:
        project_options["config_main_stack_size"] = config_main_stack_size

    project = tvm.micro.generate_project(
        str(test_utils.TEMPLATE_PROJECT_DIR),
        mod,
        temp_dir / "project",
        project_options,
    )
    project.build()
    project.flash()
    return tvm.micro.Session(project.transport())


@tvm.testing.requires_micro
@pytest.mark.skip_boards(["mps2_an521"])
def test_relay(temp_dir, board, west_cmd, tvm_debug, use_fvp):
    """Testing a simple relay graph"""

    model = test_utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": tvm_debug}
    shape = (10,)
    dtype = "int8"

    # Construct Relay program.
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)
    ir_mod = tvm.IRModule.from_expr(func)

    runtime = Runtime("crt", {"system-lib": True})
    executor = Executor("aot")
    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(ir_mod, target=target, runtime=runtime, executor=executor)

    with _make_session(temp_dir, board, west_cmd, mod, build_config, use_fvp) as session:

        aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())

        x_in = np.random.randint(10, size=shape[0], dtype=dtype)
        aot_executor.run(x=x_in)
        result = aot_executor.get_output(0).numpy()
        tvm.testing.assert_allclose(aot_executor.get_input(0).numpy(), x_in)
        tvm.testing.assert_allclose(result, x_in * x_in + 1)


@tvm.testing.requires_micro
@pytest.mark.skip_boards(["mps2_an521"])
def test_aot_executor(temp_dir, board, west_cmd, tvm_debug, use_fvp):
    """Test use of the AOT executor with microTVM."""

    model = test_utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": tvm_debug}
    shape = (10,)
    dtype = "int8"

    print("test_relay: construct relay program\n")

    # Construct Relay program.
    relay_mod = tvm.parser.fromtext(
        """
      #[version = "0.0.5"]
      def @main(%a : Tensor[(1, 2), uint8], %b : Tensor[(1, 2), uint8]) {
          %0 = %a + %b;
          %0
      }"""
    )

    runtime = Runtime("crt", {"system-lib": True})
    executor = Executor("aot")
    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(relay_mod, target=target, runtime=runtime, executor=executor)

    def do_test():

        aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())

        assert aot_executor.get_input_index("a") == 0
        assert aot_executor.get_input_index("b") == 1

        assert aot_executor.get_num_inputs() == 2
        assert aot_executor.get_num_outputs() == 1

        A_np = np.array([[2, 3]], dtype="uint8")
        B_np = np.array([[4, 7]], dtype="uint8")

        A_data = aot_executor.get_input("a").copyfrom(A_np)
        B_data = aot_executor.get_input("b").copyfrom(B_np)

        aot_executor.run()

        out = aot_executor.get_output(0)
        assert (out.numpy() == np.array([6, 10])).all()

        B_np_new = np.array([[5, 8]])
        aot_executor.set_input("b", B_np_new)
        assert (B_data.numpy() == B_np_new).all()

    with _make_session(temp_dir, board, west_cmd, mod, build_config, use_fvp) as session:
        do_test()


@tvm.testing.requires_micro
@pytest.mark.skip_boards(["mps2_an521"])
def test_strategy(temp_dir, board, west_cmd, tvm_debug, use_fvp):
    """Testing a simple relay graph"""

    model = test_utils.ZEPHYR_BOARDS[board]
    build_config = {"debug": tvm_debug}
    shape = (10,)
    dtype = "int8"

    data_shape = (4, 32, 16)
    kernel_size = 3
    num_filter = 12
    strides = 1
    padding = 0 
    dilation = 1

    ishape = data_shape
    wshape = (num_filter, data_shape[1], kernel_size)

    weight_data = np.random.randint(low=-10, high=10, size=wshape, dtype=dtype)

    input0 = relay.var("input", relay.TensorType(ishape, dtype))
    weight0 = relay.const(weight_data)
    out0 = relay.op.nn.conv1d(
        input0,
        weight0,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        data_layout="NCW",
        kernel_layout="OIW",
        out_dtype="int32",
        out_layout="NCW",
    )
    ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

    input1 = relay.var("input", relay.TensorType(ishape, dtype))
    weight1 = relay.const(weight_data)

    out1 = relay.op.nn.conv1d(
        input1,
        weight1,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        data_layout="NCW",
        kernel_layout="OIW",
        out_dtype="int32",
        out_layout="NCW",
    )
    mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
    output_list = generate_ref_data(ref_mod, inputs)

    print("INPUT", inputs)
    print("OUTPUT", output_list)

    runtime = Runtime("crt", {"system-lib": True})
    executor = Executor("aot")
    target = tvm.target.target.micro(model)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(mod, target=target, runtime=runtime, executor=executor)

    with _make_session(temp_dir, board, west_cmd, mod, build_config, use_fvp) as session:

        aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
        
        aot_executor.run(input1=inputs["input"])
        result = aot_executor.get_output(0).numpy()
        tvm.testing.assert_allclose(aot_executor.get_input(0).numpy(), inputs["input"])
        tvm.testing.assert_allclose(result, output_list["output"])

if __name__ == "__main__":
    tvm.testing.main()
