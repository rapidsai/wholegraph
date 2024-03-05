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

import os
import torch
import torch.utils.dlpack
import pylibwholegraph.binding.wholememory_binding as wmb
from .comm import set_world_info, get_global_communicator, get_local_node_communicator, reset_communicators
from .utils import str_to_wmb_wholememory_log_level


def init(world_rank: int, world_size: int, local_rank: int, local_size: int, wm_log_level="info"):
    wmb.init(0, str_to_wmb_wholememory_log_level(wm_log_level))
    set_world_info(world_rank, world_size, local_rank, local_size)


def init_torch_env(world_rank: int, world_size: int, local_rank: int, local_size: int, wm_log_level="info"):
    r"""Init WholeGraph environment for PyTorch.
    :param world_rank: world rank of current process
    :param world_size: world size of all processes
    :param local_rank: local rank of current process
    :param local_size: local size
    :return: None
    """
    os.environ["RANK"] = str(world_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if "MASTER_ADDR" not in os.environ:
        if world_rank == 0:
            print("[WARNING] MASTER_ADDR not set, resetting to localhost")
        os.environ["MASTER_ADDR"] = "localhost"

    if "MASTER_PORT" not in os.environ:
        if world_rank == 0:
            print("[WARNING] MASTER_PORT not set, resetting to 12335")
        os.environ["MASTER_PORT"] = "12335"

    wmb.init(0, str_to_wmb_wholememory_log_level(wm_log_level))
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    set_world_info(world_rank, world_size, local_rank, local_size)


def init_torch_env_and_create_wm_comm(
    world_rank: int,
    world_size: int,
    local_rank: int,
    local_size: int,
    distributed_backend_type="nccl",
    wm_log_level="info"
):
    r"""Init WholeGraph environment for PyTorch and create single communicator for all ranks.
    :param world_rank: world rank of current process
    :param world_size: world size of all processes
    :param local_rank: local rank of current process
    :param local_size: local size
    :return: global and local node Communicator
    """
    init_torch_env(world_rank, world_size, local_rank, local_size, wm_log_level)
    global_comm = get_global_communicator(distributed_backend_type)
    local_comm = get_local_node_communicator()

    return global_comm, local_comm


def finalize():
    r"""Finalize WholeGraph.
    :return: None
    """
    wmb.finalize()
    reset_communicators()
    torch.distributed.destroy_process_group() if torch.distributed.is_initialized() else None
