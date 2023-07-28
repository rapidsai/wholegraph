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

import pytest
import torch
from pylibwholegraph.test_utils.test_comm import gen_csr_graph
import pylibwholegraph.torch.graph_ops as wg_ops


def host_add_csr_self_loop(csr_row_ptr_tensor, csr_col_ptr_tensor):
    row_num = csr_row_ptr_tensor.shape[0] - 1
    edge_num = csr_col_ptr_tensor.shape[0]
    output_csr_row_ptr_tensor = torch.empty(
        (csr_row_ptr_tensor.shape[0],), dtype=csr_row_ptr_tensor.dtype
    )
    output_csr_col_ptr_tensor = torch.empty(
        (edge_num + row_num,), dtype=csr_col_ptr_tensor.dtype
    )
    for row_id in range(row_num):
        start = csr_row_ptr_tensor[row_id]
        end = csr_row_ptr_tensor[row_id + 1]
        output_csr_row_ptr_tensor[row_id] = start + row_id
        output_csr_col_ptr_tensor[start + row_id] = row_id
        for j in range(start, end):
            output_csr_col_ptr_tensor[j + row_id + 1] = csr_col_ptr_tensor[j]
    output_csr_row_ptr_tensor[row_num] = csr_row_ptr_tensor[row_num] + row_num
    return output_csr_row_ptr_tensor, output_csr_col_ptr_tensor


def routine_func(**kwargs):
    target_node_count = kwargs["target_node_count"]
    neighbor_node_count = kwargs["neighbor_node_count"]
    edge_num = kwargs["edge_num"]
    assert neighbor_node_count >= target_node_count
    csr_row_ptr_tensor, csr_col_ptr_tensor, _ = gen_csr_graph(
        target_node_count,
        edge_num,
        neighbor_node_count,
        csr_row_dtype=torch.int32,
        csr_col_dtype=torch.int32,
    )
    csr_row_ptr_tensor_cuda = csr_row_ptr_tensor.cuda()
    csr_col_ptr_tensor_cuda = csr_col_ptr_tensor.cuda()
    (
        output_csr_row_ptr_tensor_cuda,
        output_csr_col_ptr_tensor_cuda,
    ) = wg_ops.add_csr_self_loop(csr_row_ptr_tensor_cuda, csr_col_ptr_tensor_cuda)
    output_csr_row_ptr_tensor = output_csr_row_ptr_tensor_cuda.cpu()
    output_csr_col_ptr_tensor = output_csr_col_ptr_tensor_cuda.cpu()
    (
        output_csr_row_ptr_tensor_ref,
        output_csr_col_ptr_tensor_ref,
    ) = host_add_csr_self_loop(csr_row_ptr_tensor, csr_col_ptr_tensor)
    assert torch.equal(output_csr_row_ptr_tensor, output_csr_row_ptr_tensor_ref)
    assert torch.equal(output_csr_col_ptr_tensor, output_csr_col_ptr_tensor_ref)


@pytest.mark.parametrize("target_node_count", [101, 113])
@pytest.mark.parametrize("neighbor_node_count", [157, 1987])
@pytest.mark.parametrize("edge_num", [1001, 2305])
def test_add_csr_self_loop(target_node_count, neighbor_node_count, edge_num):
    gpu_count = torch.cuda.device_count()
    assert gpu_count > 0
    routine_func(
        target_node_count=target_node_count,
        neighbor_node_count=neighbor_node_count,
        edge_num=edge_num,
    )
