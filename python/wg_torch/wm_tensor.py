# Copyright (c) 2022, NVIDIA CORPORATION.
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

from enum import Enum
from typing import Union

import torch
import torch.distributed as dist

from wholegraph.torch import wholegraph_pytorch as wg


def get_unique_id():
    unique_id_array = wg.get_unique_id()
    return unique_id_array


def broadcast_unique_id_array(unique_id_array: Union[list, None], root_rank: int):
    unique_id_size_array = [0]
    if dist.get_rank() == root_rank:
        assert unique_id_array is not None
        unique_id_size_array[0] = len(unique_id_array)
    unique_id_size_tensor = torch.LongTensor(unique_id_size_array).cuda()
    dist.broadcast(unique_id_size_tensor, root_rank)
    torch.cuda.synchronize()
    unique_id_size = unique_id_size_tensor.cpu().detach().numpy()[0].item()
    if dist.get_rank() != root_rank:
        unique_id_array = [0] * unique_id_size
    unique_id_tensor = torch.LongTensor(unique_id_array).cuda()
    dist.broadcast(unique_id_tensor, root_rank)
    torch.cuda.synchronize()
    return unique_id_tensor.cpu().detach().numpy().tolist()


def broadcast_unique_id_array_by_continuous_group(
    unique_id_array: Union[list, None], group_root_rank: int, group_size: int
):
    size = dist.get_world_size()
    rank = dist.get_rank()
    assert group_size >= 1 and size % group_size == 0
    assert group_root_rank < group_size
    group_count = size // group_size
    group_unique_id = None
    for gidx in range(group_count):
        recv_id_array = broadcast_unique_id_array(
            unique_id_array, gidx * group_size + group_root_rank
        )
        if gidx * group_size <= rank < (gidx + 1) * group_size:
            group_unique_id = recv_id_array
    return group_unique_id


def create_communicator(size: int, unique_id_array: list, rank: int):
    return wg.create_communicator(size, unique_id_array, rank)


def create_intra_node_communicator(world_rank, world_size, local_size):
    assert world_size % local_size == 0
    unique_id_array = None
    if world_rank % local_size == 0:
        print(
            "creating_intra_node_communicator root=%d, local_size=%d, world_size=%d"
            % (world_rank, local_size, world_size)
        )
        unique_id_array = get_unique_id()
    unique_id_array = broadcast_unique_id_array_by_continuous_group(
        unique_id_array, 0, local_size
    )
    return create_communicator(local_size, unique_id_array, world_rank % local_size)


def create_global_communicator(world_rank, world_size):
    unique_id_array = None
    if world_rank == 0:
        unique_id_array = get_unique_id()
    unique_id_array = broadcast_unique_id_array(unique_id_array, 0)
    return create_communicator(world_size, unique_id_array, world_rank)


def get_wm_communicator(t: Union[torch.Tensor, wg.ChunkedTensor, wg.NCCLTensor]):
    if isinstance(t, torch.Tensor):
        return wg.get_tensor_communicator(t)
    elif isinstance(t, wg.ChunkedTensor):
        return wg.get_chunked_tensor_communicator(t)
    else:
        assert isinstance(t, wg.NCCLTensor)
        return wg.get_nccl_tensor_communicator(t)


def get_partition_plan(
    tc: Union[torch.Tensor, wg.ChunkedTensor, wg.NCCLTensor, int],
    total_entry_count: Union[int, None] = None,
):
    if isinstance(tc, int):
        comm = tc
        assert total_entry_count is not None
    else:
        comm = get_wm_communicator(tc)
        tensor_dim_0 = tc.shape[0]
        if total_entry_count is not None:
            assert total_entry_count == tensor_dim_0
        else:
            total_entry_count = tensor_dim_0
    size = wg.get_size(comm)
    rank = wg.get_rank(comm)
    entry_per_rank = (total_entry_count + size - 1) // size
    local_start = rank * entry_per_rank
    local_end = min(entry_per_rank * (rank + 1), total_entry_count)
    local_entry_count = local_end - local_start
    return local_start, local_entry_count, entry_per_rank, total_entry_count


def get_local_tensor(t: Union[torch.Tensor, wg.ChunkedTensor, wg.NCCLTensor]):
    # split along dim 0
    assert len(t.shape) >= 1
    if isinstance(t, torch.Tensor):
        start_dim_0, dim_0_count, _, _ = get_partition_plan(t)
        end_dim_0 = start_dim_0 + dim_0_count
        return t[start_dim_0:end_dim_0]
    elif isinstance(t, wg.ChunkedTensor):
        return wg.get_local_tensor_from_chunked_tensor(t, 0)
    elif isinstance(t, wg.NCCLTensor):
        return wg.get_local_tensor_from_nccl_tensor(t, 0)
    else:
        raise TypeError("Tensor type not supported.")


class WmTensorType(Enum):
    HOST = 0
    DEVICE = 1
    CHUNKED = 2
    NCCL = 3


def get_intra_node_wm_tensor_type(chunked=True, use_host_memory=False):
    if chunked:
        return WmTensorType.CHUNKED
    else:
        if use_host_memory:
            return WmTensorType.HOST
        else:
            return WmTensorType.DEVICE


def get_wm_tensor_type(t: Union[torch.Tensor, wg.ChunkedTensor, wg.NCCLTensor]):
    if isinstance(t, torch.Tensor):
        if wg.is_unified_tensor(t):
            return WmTensorType.HOST
        else:
            return WmTensorType.DEVICE
    elif isinstance(t, wg.ChunkedTensor):
        return WmTensorType.CHUNKED
    else:
        assert isinstance(t, wg.NCCLTensor)
        return WmTensorType.NCCL


def create_wm_tensor(
    wm_comm,
    sizes,
    strides=[],
    tensor_dtype=torch.float32,
    wm_tensor_type: WmTensorType = WmTensorType.CHUNKED,
):
    if wm_tensor_type == WmTensorType.CHUNKED:
        return wg.create_chunked_tensor(sizes, strides, tensor_dtype, wm_comm)
    elif wm_tensor_type == WmTensorType.HOST:
        return wg.get_tensor_view(
            wg.create_tensor(sizes, strides, tensor_dtype, True, wm_comm),
            torch.device("cuda", torch.cuda.current_device()),
        )
    elif wm_tensor_type == WmTensorType.DEVICE:
        return wg.create_tensor(sizes, strides, tensor_dtype, False, wm_comm)
    elif wm_tensor_type == WmTensorType.NCCL:
        return wg.create_nccl_tensor(sizes, strides, tensor_dtype, wm_comm)
    else:
        raise ValueError("Invalid wm_tensor_type")


def create_wm_tensor_from_file(
    shape,
    dtype,
    wm_comm,
    filename,
    wm_tensor_type: WmTensorType = WmTensorType.CHUNKED,
    part_count: int = 0,
):
    file_elt_count = wg.stat_filelist_element_count(filename, dtype)
    if len(shape) != 0:
        shape_count = 1
        for shape_dimsize in shape:
            shape_count *= shape_dimsize
        assert shape_count == file_elt_count
    else:
        shape = (file_elt_count,)
    wmt = create_wm_tensor(wm_comm, shape, [], dtype, wm_tensor_type)
    lt = get_local_tensor(wmt)
    lt_2d = lt.reshape((lt.numel(), 1))
    wg.load_local_tensor_from_embedding_file(lt_2d, filename, part_count, wm_comm)
    return wmt
