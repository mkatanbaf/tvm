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
"""
.. _microTVM-with-TFLite:

microTVM with TFLite Models
===========================
**Author**: `Tom Gall <https://github.com/tom-gall>`_

This tutorial is an introduction to working with microTVM and a TFLite
model with Relay.
"""

import os
import json
import tarfile
import pathlib
import tempfile
import numpy as np

import tvm
from tvm import relay
import tvm.contrib.utils
from tvm.contrib.download import download_testdata
from tvm.relay.backend import Runtime

use_physical_hw = False; #bool(os.getenv("TVM_MICRO_USE_HW"))

# model_url = "https://people.linaro.org/~tom.gall/sine_model.tflite"
# model_file = "sine_model.tflite"
# model_path = download_testdata(model_url, model_file, module="data")
# tflite_model_buf = open(model_path, "rb").read()

model_file = "example_model.tflite"
model_path = "./"+model_file
tflite_model_buf = open(model_path, "rb").read()

######################################################################
# Using the buffer, transform into a tflite model python object
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Print out the version of the model
version = tflite_model.Version()
print("Model Version: " + str(version))

######################################################################
# Parse the python model object to convert it into a relay module
# and weights.
# It is important to note that the input tensor name must match what
# is contained in the model.
#
# If you are unsure what that might be, this can be discovered by using
# the ``visualize.py`` script within the Tensorflow project.
# See `How do I inspect a .tflite file? <https://www.tensorflow.org/lite/guide/faq>`_

input_tensor = "dense_4_input"
input_shape = (1,)
input_dtype = "float32"

relay_mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

######################################################################
# Defining the target
# -------------------
#
# Now we create a build config for relay, turning off two options and then calling relay.build which
# will result in a C source file for the selected TARGET. When running on a simulated target of the
# same architecture as the host (where this Python script is executed) choose "host" below for the
# TARGET, the C Runtime as the RUNTIME and a proper board/VM to run it (Zephyr will create the right
# QEMU VM based on BOARD. In the example below the x86 arch is selected and a x86 VM is picked up accordingly:
#
RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})
TARGET = tvm.target.target.micro("host")

#
# Compiling for physical hardware
#  When running on physical hardware, choose a TARGET and a BOARD that describe the hardware. The
#  STM32F746 Nucleo target and board is chosen in the example below. Another option would be to
#  choose the STM32F746 Discovery board instead. Since that board has the same MCU as the Nucleo
#  board but a couple of wirings and configs differ, it's necessary to select the "stm32f746g_disco"
#  board to generated the right firmware image.
#

if use_physical_hw:
    boards_file = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr")) / "boards.json"
    with open(boards_file) as f:
        boards = json.load(f)

    BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_l4r5zi")
    TARGET = tvm.target.target.micro(boards[BOARD]["model"])





#########################
# Extracting tuning tasks
#########################
# Not all operators in the Relay program printed above can be tuned. Some are so trivial that only
# a single implementation is defined; others don't make sense as tuning tasks. Using
# `extract_from_program`, you can produce a list of tunable tasks.
#
# Because task extraction involves running the compiler, we first configure the compiler's
# transformation passes; we'll apply the same configuration later on during autotuning.
#

pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
with pass_context:
    tasks = tvm.autotvm.task.extract_from_program(relay_mod["main"], {}, TARGET)
assert len(tasks) > 0

######################
# Configuring microTVM
######################
# Before autotuning, we need to define a module loader and then pass that to
# a `tvm.autotvm.LocalBuilder`. Then we create a `tvm.autotvm.LocalRunner` and use
# both builder and runner to generates multiple measurements for auto tunner.
#
# In this tutorial, we have the option to use x86 host as an example or use different targets
# from Zephyr RTOS. If you choose pass `--platform=host` to this tutorial it will uses x86. You can
# choose other options by choosing from `PLATFORM` list.
#

module_loader = tvm.micro.AutoTvmModuleLoader(
    template_project_dir=pathlib.Path(tvm.micro.get_microtvm_template_projects("crt")),
    project_options={"verbose": False},
)
builder = tvm.autotvm.LocalBuilder(
    n_parallel=1,
    build_kwargs={"build_option": {"tir.disable_vectorize": True}},
    do_fork=True,
    build_func=tvm.micro.autotvm_build_func,
    runtime=RUNTIME,
)
runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)

measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

# Compiling for physical hardware
if use_physical_hw:
    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr")),
        project_options={
            "zephyr_board": BOARD,
            "west_cmd": "west",
            "verbose": False,
            "project_type": "host_driven",
        },
    )
    builder = tvm.autotvm.LocalBuilder(
        n_parallel=1,
        build_kwargs={"build_option": {"tir.disable_vectorize": True}},
        do_fork=False,
        build_func=tvm.micro.autotvm_build_func,
        runtime=RUNTIME,
    )
    runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)

    measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

##########################
# Run Autotuning
##########################
# Now we can run autotuning separately on each extracted task on microTVM device.
#

autotune_log_file = pathlib.Path("microtvm_autotune.log.txt")
if os.path.exists(autotune_log_file):
    os.remove(autotune_log_file)

num_trials = 10
for task in tasks:
    tuner = tvm.autotvm.tuner.GATuner(task)
    tuner.tune(
        n_trial=num_trials,
        measure_option=measure_option,
        callbacks=[
            tvm.autotvm.callback.log_to_file(str(autotune_log_file)),
            tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M"),
        ],
        si_prefix="M",
    )

############################
# Timing the untuned program
############################
# For comparison, let's compile and run the graph without imposing any autotuning schedules. TVM
# will select a randomly-tuned implementation for each operator, which should not perform as well as
# the tuned operator.
#

with pass_context:
    lowered = tvm.relay.build(relay_mod, target=TARGET, runtime=RUNTIME, params=params)

temp_dir = tvm.contrib.utils.tempdir()
project = tvm.micro.generate_project(
    str(tvm.micro.get_microtvm_template_projects("crt")),
    lowered,
    temp_dir / "project",
    {"verbose": False},
)

# Compiling for physical hardware
if use_physical_hw:
    temp_dir = tvm.contrib.utils.tempdir()
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects("zephyr")),
        lowered,
        temp_dir / "project",
        {
            "zephyr_board": BOARD,
            "west_cmd": "west",
            "verbose": False,
            "project_type": "host_driven",
        },
    )

project.build()
project.flash()
with tvm.micro.Session(project.transport()) as session:
    debug_module = tvm.micro.create_local_debug_executor(
        lowered.get_graph_json(), session.get_system_lib(), session.device
    )
    debug_module.set_input(**lowered.get_params())
    print("########## Build without Autotuning ##########")
    debug_module.run()
    del debug_module

##########################
# Timing the tuned program
##########################
# Once autotuning completes, you can time execution of the entire program using the Debug Runtime:

with tvm.autotvm.apply_history_best(str(autotune_log_file)):
    with pass_context:
        lowered_tuned = tvm.relay.build(relay_mod, target=TARGET, runtime=RUNTIME, params=params)

temp_dir = tvm.contrib.utils.tempdir()
project = tvm.micro.generate_project(
    str(tvm.micro.get_microtvm_template_projects("crt")),
    lowered_tuned,
    temp_dir / "project",
    {"verbose": False},
)

# Compiling for physical hardware
if use_physical_hw:
    temp_dir = tvm.contrib.utils.tempdir()
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects("zephyr")),
        lowered_tuned,
        temp_dir / "project",
        {
            "zephyr_board": BOARD,
            "west_cmd": "west",
            "verbose": False,
            "project_type": "host_driven",
        },
    )

project.build()
project.flash()
with tvm.micro.Session(project.transport()) as session:
    debug_module = tvm.micro.create_local_debug_executor(
        lowered_tuned.get_graph_json(), session.get_system_lib(), session.device
    )
    debug_module.set_input(**lowered_tuned.get_params())
    print("########## Build with Autotuning ##########")
    debug_module.run()
    del debug_module

















# #
# #  For some boards, Zephyr runs them emulated by default, using QEMU. For example, below is the
# #  TARGET and BOARD used to build a microTVM firmware for the mps2-an521 board. Since that board
# #  runs emulated by default on Zephyr the suffix "-qemu" is added to the board name to inform
# #  microTVM that the QEMU transporter must be used to communicate with the board. If the board name
# #  already has the prefix "qemu_", like "qemu_x86", then it's not necessary to add that suffix.
# #
# #  TARGET = tvm.target.target.micro("mps2_an521")
# #  BOARD = "mps2_an521-qemu"

# ######################################################################
# # Now, compile the model for the target:

# with tvm.transform.PassContext(
#     opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["AlterOpLayout"]
# ):
#     module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=params)


# # Inspecting the compilation output
# # ---------------------------------
# #
# # The compilation process has produced some C code implementing the operators in this graph. We
# # can inspect it by printing the CSourceModule contents (for the purposes of this tutorial, let's
# # just print the first 10 lines):

# c_source_module = module.get_lib().imported_modules[0]
# assert c_source_module.type_key == "c", "tutorial is broken"

# c_source_code = c_source_module.get_source()
# first_few_lines = c_source_code.split("\n")[:]
# assert any(
#     l.startswith("TVM_DLL int32_t tvmgen_default_") for l in first_few_lines
# ), f"tutorial is broken: {first_few_lines!r}"
# print("\n".join(first_few_lines))


# # Compiling the generated code
# # ----------------------------
# #
# # Now we need to incorporate the generated C code into a project that allows us to run inference on the
# # device. The simplest way to do this is to integrate it yourself, using microTVM's standard output format
# # (:doc:`Model Library Format` </dev/model_library_format>`). This is a tarball with a standard layout:

# # Get a temporary path where we can store the tarball (since this is running as a tutorial).

# fd, model_library_format_tar_path = tempfile.mkstemp()
# print("DIR : ", model_library_format_tar_path)
# os.close(fd)
# os.unlink(model_library_format_tar_path)
# tvm.micro.export_model_library_format(module, model_library_format_tar_path)

# with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
#     print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))

# # Cleanup for tutorial:
# os.unlink(model_library_format_tar_path)


# # TVM also provides a standard way for embedded platforms to automatically generate a standalone
# # project, compile and flash it to a target, and communicate with it using the standard TVM RPC
# # protocol. The Model Library Format serves as the model input to this process. When embedded
# # platforms provide such an integration, they can be used directly by TVM for both host-driven
# # inference and autotuning . This integration is provided by the
# # `microTVM Project API` <https://github.com/apache/tvm-rfcs/blob/main/rfcs/0008-microtvm-project-api.md>_,
# #
# # Embedded platforms need to provide a Template Project containing a microTVM API Server (typically,
# # this lives in a file ``microtvm_api_server.py`` in the root directory). Let's use the example ``host``
# # project in this tutorial, which simulates the device using a POSIX subprocess and pipes:

# template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
# project_options = {}  # You can use options to provide platform-specific options through TVM.

# # Compiling for physical hardware (or an emulated board, like the mps_an521)
# # --------------------------------------------------------------------------
# #  For physical hardware, you can try out the Zephyr platform by using a different template project
# #  and options:
# #

# if use_physical_hw:
#     template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
#     project_options = {"project_type": "host_driven", "zephyr_board": BOARD}

# # Create a temporary directory

# temp_dir = tvm.contrib.utils.tempdir()
# generated_project_dir = temp_dir / "generated-project"
# generated_project = tvm.micro.generate_project(
#     template_project_path, module, generated_project_dir, project_options
# )

# # Build and flash the project
# generated_project.build()
# generated_project.flash()


# ######################################################################
# # Next, establish a session with the simulated device and run the
# # computation. The `with session` line would typically flash an attached
# # microcontroller, but in this tutorial, it simply launches a subprocess
# # to stand in for an attached microcontroller.

# with tvm.micro.Session(transport_context_manager=generated_project.transport()) as session:
#     graph_mod = tvm.micro.create_local_graph_executor(
#         module.get_graph_json(), session.get_system_lib(), session.device
#     )

#     # Set the model parameters using the lowered parameters produced by `relay.build`.
#     graph_mod.set_input(**module.get_params())

#     # The model consumes a single float32 value and returns a predicted sine value.  To pass the
#     # input value we construct a tvm.nd.array object with a single contrived number as input. For
#     # this model values of 0 to 2Pi are acceptable.
#     graph_mod.set_input(input_tensor, tvm.nd.array(np.array([0.5], dtype="float32")))
#     graph_mod.run()

#     tvm_output = graph_mod.get_output(0).numpy()
#     print("result is: " + str(tvm_output))
