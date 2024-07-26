# Copyright (c) 2024, NVIDIA CORPORATION.
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
import numpy as np
import pylibwholegraph.binding.wholememory_binding as wmb
from pylibwholegraph.torch.dlpack_utils import torch_import_from_dlpack
from packaging import version


def gen_csr_format_from_dense_matrix(
    matrix_tensor,
    graph_node_count,
    graph_edge_count,
    neighbor_node_count,
    csr_row_dtype,
    csr_col_dtype,
    weight_dtype,
):
    row_num = matrix_tensor.shape[0]
    col_num = matrix_tensor.shape[1]
    assert row_num == graph_node_count
    assert col_num == neighbor_node_count
    csr_row_ptr = torch.zeros((graph_node_count + 1,), dtype=csr_row_dtype)
    for i in range(row_num):
        csr_row_ptr[i + 1] = torch.count_nonzero(matrix_tensor[i]).item()
    csr_row_ptr = torch.cumsum(csr_row_ptr, dim=0, dtype=csr_row_dtype)
    assert csr_row_ptr[graph_node_count] == graph_edge_count
    csr_col_ptr = torch.nonzero(matrix_tensor, as_tuple=True)[1]
    csr_weight_ptr = torch.empty((graph_edge_count,), dtype=weight_dtype)
    for row_id in range(row_num):
        start = csr_row_ptr[row_id]
        end = csr_row_ptr[row_id + 1]
        for j in range(start, end):
            col_id = csr_col_ptr[j]
            csr_weight_ptr[j] = matrix_tensor[row_id][col_id]

    if csr_col_dtype == torch.int32:
        csr_col_ptr = csr_col_ptr.int()

    return csr_row_ptr, csr_col_ptr, csr_weight_ptr


def gen_csr_graph(
    graph_node_count,
    graph_edge_count,
    neighbor_node_count=None,
    csr_row_dtype=torch.int64,
    csr_col_dtype=torch.int32,
    weight_dtype=torch.float32,
):
    if neighbor_node_count is None:
        neighbor_node_count = graph_node_count
    all_count = graph_node_count * neighbor_node_count
    assert all_count >= graph_edge_count
    matrix_tensor = (
        torch.rand(all_count, dtype=weight_dtype, device=torch.device("cpu")) + 1
    )
    choice_zero_idxs = torch.randperm(all_count, device=torch.device("cpu"))[
        : all_count - graph_edge_count
    ]
    matrix_tensor[choice_zero_idxs] = 0
    matrix_tensor.resize_(graph_node_count, neighbor_node_count)
    target_torch_version = "1.13.0a"

    if version.parse(torch.__version__) >= version.parse(target_torch_version):
        sp_format = matrix_tensor.to_sparse_csr()
        csr_row_ptr = sp_format.crow_indices()
        csr_col_ptr = sp_format.col_indices()
        csr_weight_ptr = sp_format.values()
        assert csr_row_ptr.dtype == torch.int64
        assert csr_col_ptr.dtype == torch.int64
        if csr_col_dtype == torch.int32:
            csr_col_ptr = csr_col_ptr.int()
        if csr_row_dtype == torch.int32:
            csr_row_ptr = csr_row_ptr.int()
        return csr_row_ptr, csr_col_ptr, csr_weight_ptr
    else:
        return gen_csr_format_from_dense_matrix(
            matrix_tensor,
            graph_node_count,
            graph_edge_count,
            neighbor_node_count,
            csr_row_dtype,
            csr_col_dtype,
            weight_dtype,
        )


def host_sample_all_neighbors(
    host_csr_row_ptr,
    host_csr_col_ptr,
    center_nodes,
    output_sample_offset_tensor,
    col_id_dtype,
    total_sample_count,
):
    output_dest_tensor = torch.empty((total_sample_count,), dtype=col_id_dtype)
    output_center_localid_tensor = torch.empty((total_sample_count,), dtype=torch.int32)
    output_edge_gid_tensor = torch.empty((total_sample_count,), dtype=torch.int64)
    center_nodes_count = center_nodes.size(0)

    for i in range(center_nodes_count):
        node_id = center_nodes[i]
        start = host_csr_row_ptr[node_id]
        end = host_csr_row_ptr[node_id + 1]
        output_id = output_sample_offset_tensor[i]
        for j in range(end - start):
            output_dest_tensor[output_id + j] = host_csr_col_ptr[start + j]
            output_center_localid_tensor[output_id + j] = i
            output_edge_gid_tensor[output_id + j] = start + j
    return output_sample_offset_tensor, output_dest_tensor, output_center_localid_tensor, output_edge_gid_tensor


def copy_host_1D_tensor_to_wholememory(
    wm_array, host_tensor, world_rank, world_size, wm_comm
):

    local_tensor_cuda, local_start = wm_array.get_local_tensor(
        torch_import_from_dlpack, wmb.WholeMemoryMemoryLocation.MlDevice, world_rank
    )
    assert local_tensor_cuda.dim() == 1
    local_count = wm_array.get_local_entry_count()
    local_start_ref = wm_array.get_local_entry_start()
    assert local_start == local_start_ref
    assert local_tensor_cuda.shape[0] == local_count
    local_tensor_cuda.copy_(host_tensor[local_start : local_start + local_count])
    wm_comm.barrier()


def host_get_sample_offset_tensor(host_csr_row_ptr, center_nodes, max_sample_count):
    center_nodes_count = center_nodes.size(0)
    output_sample_offset_tensor = torch.empty(
        (center_nodes_count + 1,), dtype=torch.int32
    )
    output_sample_offset_tensor[0] = 0
    for i in range(center_nodes_count):
        node_id = center_nodes[i]
        neighbor_count = host_csr_row_ptr[node_id + 1] - host_csr_row_ptr[node_id]
        output_sample_offset_tensor[i + 1] = neighbor_count
        if max_sample_count > 0:
            output_sample_offset_tensor[i + 1] = min(max_sample_count, neighbor_count)
    output_sample_offset_tensor = torch.cumsum(
        output_sample_offset_tensor, dim=0, dtype=torch.int32
    )
    return output_sample_offset_tensor


def int_to_wholememory_datatype(value: int):
    if value == 0:
        return wmb.WholeMemoryDataType.DtInt
    if value == 1:
        return wmb.WholeMemoryDataType.DtInt64
    if value == 2:
        return wmb.WholeMemoryDataType.DtFloat
    if value == 3:
        return wmb.WholeMemoryDataType.DtDouble
    else:
        raise ValueError("invalid int_to_wholememory_datatype value")


def int_to_wholememory_location(value: int):
    if value == 0:
        return wmb.WholeMemoryMemoryLocation.MlHost
    if value == 1:
        return wmb.WholeMemoryMemoryLocation.MlDevice
    else:
        raise ValueError("invalid int_to_wholememory_localtion value")


def int_to_wholememory_type(value: int):
    if value == 0:
        return wmb.WholeMemoryMemoryType.MtContinuous
    if value == 1:
        return wmb.WholeMemoryMemoryType.MtChunked
    if value == 2:
        return wmb.WholeMemoryMemoryType.MtDistributed
    if value == 3:
        return wmb.WholeMemoryMemoryType.MtHierarchy
    else:
        raise ValueError("invalid int_to_wholememory_type value")


def random_partition(total_entry_count: int, world_size: int) -> np.array:
    np.random.seed(42)
    random_array = np.random.uniform(90, 100, size=world_size)
    random_sum = np.sum(random_array)
    partition = ((random_array / random_sum) * total_entry_count).astype(np.uintp)
    diff = total_entry_count - np.sum(partition)
    partition[0] += diff
    return partition
