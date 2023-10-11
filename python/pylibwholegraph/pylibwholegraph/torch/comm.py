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

import torch
import torch.distributed as dist
import torch.utils.dlpack
import pylibwholegraph.binding.wholememory_binding as wmb


global_communicator = None
local_node_communicator = None
local_device_communicator = None

all_comm_world_rank = 0
all_comm_world_size = 1
all_comm_local_rank = 0
all_comm_local_size = 1


def set_world_info(world_rank: int, world_size: int, local_rank: int, local_size: int):
    """
    Set the global world's information. This is used for create common used communicators, like local node communicator,
    global communicator, or local device communicator.

    :param world_rank: world rank of current process.
    :param world_size: world size
    :param local_rank: local rank of current process in current machine node.
    :param local_size: local size of each machine node
    :return: None
    """
    global all_comm_world_rank, all_comm_world_size, all_comm_local_rank, all_comm_local_size
    all_comm_world_rank = world_rank
    all_comm_world_size = world_size
    all_comm_local_rank = local_rank
    all_comm_local_size = local_size


class WholeMemoryCommunicator(object):
    """
    WholeMemory Communicator.
    You should not create object of this class directly, use create_group_communicator, get_global_communicator,
    get_local_node_communicator or get_local_device_communicator instead.
    """

    def __init__(self, wmb_comm: wmb.PyWholeMemoryComm):
        super().__init__()
        self.wmb_comm = wmb_comm

    def get_rank(self):
        """Get rank of current process in this communicator"""
        return self.wmb_comm.get_rank()

    def get_size(self):
        """Get world size of this communicator"""
        return self.wmb_comm.get_size()

    def barrier(self):
        """
        Barrier on WholeMemory Communicator.
        This function will use internal communicator associated CUDA stream. And synchronized with host.
        So if you have work in other CUDA stream, and expect that to be done before barrier, you may need to
        synchrionze that stream before calling this function.
        """
        return self.wmb_comm.barrier()

    def destroy(self):
        wmb.destroy_communicator(self.wmb_comm)
        self.wmb_comm = None


def create_group_communicator(group_size: int = -1, comm_stride: int = 1):
    """Create WholeMemory Communicator.
    For example: 24 ranks with group_size = 4 and comm_stride = 2 will create following groups:
    [0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15], [16, 18, 20, 22], [17, 19, 21, 23]
    :param group_size: Size of each group, -1 means to use all ranks in just one single group.
    :param comm_stride: Stride of each rank in each group
    :return: WholeMemoryCommunicator
    """
    world_size = dist.get_world_size()
    if group_size == -1:
        group_size = world_size
    strided_group_size = group_size * comm_stride
    assert world_size % strided_group_size == 0
    strided_group_count = world_size // strided_group_size
    world_rank = dist.get_rank()
    strided_group_idx = world_rank // strided_group_size
    idx_in_strided_group = world_rank % strided_group_size
    inner_group_idx = idx_in_strided_group % comm_stride
    idx_in_group = idx_in_strided_group // comm_stride
    wm_uid = wmb.PyWholeMemoryUniqueID()
    for strided_group in range(strided_group_count):
        for inner_group in range(comm_stride):
            group_root_rank = strided_group * strided_group_size + inner_group
            if world_rank == group_root_rank:
                tmp_wm_uid = wmb.create_unique_id()
            else:
                tmp_wm_uid = wmb.PyWholeMemoryUniqueID()
            uid_th = torch.utils.dlpack.from_dlpack(tmp_wm_uid.__dlpack__())
            uid_th_cuda = uid_th.cuda()
            dist.broadcast(uid_th_cuda, group_root_rank)
            uid_th.copy_(uid_th_cuda.cpu())
            if strided_group_idx == strided_group and inner_group_idx == inner_group:
                wm_uid_th = torch.utils.dlpack.from_dlpack(wm_uid.__dlpack__())
                wm_uid_th.copy_(uid_th)
    wm_comm = wmb.create_communicator(wm_uid, idx_in_group, group_size)
    return WholeMemoryCommunicator(wm_comm)


def destroy_communicator(wm_comm: WholeMemoryCommunicator):
    """
    Destroy WholeMemoryCommunicator
    :param wm_comm: WholeMemoryCommunicator to destroy
    :return: None
    """
    if wm_comm is not None and wm_comm.wmb_comm is not None:
        wmb.destroy_communicator(wm_comm.wmb_comm)
        wm_comm.wmb_comm = None


def get_global_communicator():
    """
    Get the global communicator of this job
    :return: WholeMemoryCommunicator that has all GPUs in it.
    """
    global global_communicator, local_node_communicator, local_device_communicator
    global all_comm_local_size, all_comm_world_size
    if global_communicator is None:
        global_communicator = create_group_communicator()
        if all_comm_local_size == all_comm_world_size:
            assert local_node_communicator is None
            local_node_communicator = global_communicator
        if all_comm_world_size == 1:
            assert local_device_communicator is None
            local_device_communicator = global_communicator
    return global_communicator


def get_local_node_communicator():
    """
    Get the local node communicator of this job
    :return: WholeMemoryCommunicator that has GPUs in the same node.
    """
    global global_communicator, local_node_communicator, local_device_communicator
    global all_comm_local_size, all_comm_world_size
    if local_node_communicator is None:
        local_node_communicator = create_group_communicator(all_comm_local_size)
        if all_comm_local_size == all_comm_world_size:
            assert global_communicator is None
            global_communicator = local_node_communicator
        if all_comm_local_size == 1:
            assert local_device_communicator is None
            local_device_communicator = local_node_communicator
    return local_node_communicator


def get_local_device_communicator():
    """
    Get the local device communicator of this job
    :return: WholeMemoryCommunicator that has only the GPU belonging to current process.
    """
    global global_communicator, local_node_communicator, local_device_communicator
    global all_comm_local_size, all_comm_world_size
    if local_device_communicator is None:
        local_device_communicator = create_group_communicator(1)
        if all_comm_local_size == 1:
            assert local_node_communicator is None
            local_node_communicator = local_device_communicator
        if all_comm_world_size == 1:
            assert global_communicator is None
            global_communicator = local_device_communicator
    return local_device_communicator

def init_nvshmem_with_comm(wm_comm: WholeMemoryCommunicator):
    """
    Init nvshmem backend with a WholeMemoryCommunicator, the function must be called before creating wholememory of nvshmem backend. 
    :param wm_comm: WholeMemoryCommunicator to destroy
    :return: None
    """
    wmb.init_nvshmem_with_communicator(wm_comm.wmb_comm)
    return

def finalize_nvshmem_with_comm(wm_comm: WholeMemoryCommunicator):
    """
    Finalize nvshmem backend with a WholeMemoryCommunicator, the wm_comm must be the one that be used to init nvshmem backend.
    :param wm_comm: WholeMemoryCommunicator to destroy
    :return: None
    """
    wmb.finalize_nvshmem_with_communicator(wm_comm.wmb_comm)
    return