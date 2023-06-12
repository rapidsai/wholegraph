# Copyright (c) 2019-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp


def multiprocess_run(world_size: int, func, inline_single_process=False):
    """
    Run func in multiple process
    :param world_size: process count
    :param func: function to run
    :param inline_single_process: when only one process, whether to use current process to run.
    :return: None
    """
    assert world_size > 0
    if world_size == 1 and inline_single_process:
        func(0, 1)
        return
    spawn_context = mp.get_context("spawn")

    process_array = [None] * world_size
    for i in range(world_size):
        process_array[i] = spawn_context.Process(target=func, args=(i, world_size))
        process_array[i].start()
    for i in range(world_size):
        process_array[i].join()
    for i in range(world_size):
        assert process_array[i].exitcode == 0
