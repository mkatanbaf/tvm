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
import datetime
import os
import pathlib

import pytest

import test_utils

from tvm.contrib.utils import tempdir


def pytest_addoption(parser):
    parser.addoption(
        "--zephyr-board",
        required=True,
        choices=test_utils.ZEPHYR_BOARDS.keys(),
        help="Zephyr board for test.",
    )
    parser.addoption(
        "--west-cmd", default="west", help="Path to `west` command for flashing device."
    )
    parser.addoption(
        "--tvm-debug",
        action="store_true",
        default=False,
        help="If set true, enable a debug session while the test is running. Before running the test, in a separate shell, you should run: <python -m tvm.exec.microtvm_debug_shell>",
    )
    parser.addoption(
        "--use-fvp",
        action="store_true",
        default=False,
        help="If set true, use the FVP emulator to run the test",
    )


def pytest_generate_tests(metafunc):
    if "board" in metafunc.fixturenames:
        metafunc.parametrize("board", [metafunc.config.getoption("zephyr_board")])


@pytest.fixture
def west_cmd(request):
    return request.config.getoption("--west-cmd")


@pytest.fixture
def tvm_debug(request):
    return request.config.getoption("--tvm-debug")


@pytest.fixture
def use_fvp(request):
    return request.config.getoption("--use-fvp")


@pytest.fixture
def temp_dir(board, tvm_debug):
    parent_dir = pathlib.Path(os.path.dirname(__file__))
    filename = os.path.splitext(os.path.basename(__file__))[0]
    board_workspace = (
        parent_dir
        / f"workspace_{filename}_{board}"
        / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )
    board_workspace_base = str(board_workspace)
    number = 1
    while board_workspace.exists():
        board_workspace = pathlib.Path(board_workspace_base + f"-{number}")
        number += 1

    if not os.path.exists(board_workspace.parent):
        os.makedirs(board_workspace.parent)

    keep_for_debug = tvm_debug if tvm_debug else None
    test_temp_dir = tempdir(custom_path=board_workspace, keep_for_debug=keep_for_debug)
    return test_temp_dir


@pytest.fixture(autouse=True)
def skip_by_board(request, board):
    """Skip test if board is in the list."""
    if request.node.get_closest_marker("skip_boards"):
        if board in request.node.get_closest_marker("skip_boards").args[0]:
            pytest.skip("skipped on this board: {}".format(board))


@pytest.fixture(autouse=True)
def xfail_by_platform(request):
    """mark the tests as xfail if running on fvp."""
    if request.node.get_closest_marker("xfail_on_fvp"):
        request.node.add_marker(pytest.mark.xfail(reason="checking corstone300 reliability on CI"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_boards(board): skip test for the given board",
    )
    config.addinivalue_line(
        "markers",
        "xfail_by_platform(): mark test as xfail on fvp",
    )
