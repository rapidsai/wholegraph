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
from pylibwholegraph.torch.initialize import load_wholegraph_op_libraries
import torch
from pylibwholegraph.test_utils.test_comm import gen_csr_graph
import pylibwholegraph.torch.graph_ops as wg_ops


def host_gspmm_weighted_forward(
    csr_row_ptr_tensor, csr_col_ptr_tensor, edge_weight_tensor, feature_tensor
):
    row_num = csr_row_ptr_tensor.shape[0] - 1
    # col_num = feature_tensor.shape[0]
    # edge_num = csr_col_ptr_tensor.shape[0]
    num_head = edge_weight_tensor.shape[1]
    feature_dim = feature_tensor.shape[2]
    output_feature_tensor = torch.zeros(
        (row_num, num_head, feature_dim), dtype=feature_tensor.dtype
    )

    for row_id in range(row_num):
        start = csr_row_ptr_tensor[row_id]
        end = csr_row_ptr_tensor[row_id + 1]
        for head_id in range(num_head):
            for feature_id in range(feature_dim):
                for j in range(start, end):
                    col_id = csr_col_ptr_tensor[j]
                    output_feature_tensor[row_id][head_id][feature_id] += (
                        edge_weight_tensor[j][head_id]
                        * feature_tensor[col_id][head_id][feature_id]
                    )

    return output_feature_tensor


def routine_func(**kwargs):
    load_wholegraph_op_libraries()
    target_node_count = kwargs["target_node_count"]
    neighbor_node_count = kwargs["neighbor_node_count"]
    edge_num = kwargs["edge_num"]
    feature_dtype = kwargs["feature_dtype"]
    feature_dim = kwargs["feature_dim"]
    num_head = kwargs["num_head"]
    assert neighbor_node_count >= target_node_count
    csr_row_ptr_tensor, csr_col_ptr_tensor, _ = gen_csr_graph(
        target_node_count,
        edge_num,
        neighbor_node_count,
        csr_row_dtype=torch.int32,
        csr_col_dtype=torch.int32,
    )
    edge_weight_tensor = torch.rand((edge_num, num_head), dtype=feature_dtype)
    feature_tensor = torch.rand(
        (neighbor_node_count, num_head, feature_dim), dtype=feature_dtype
    )
    csr_row_ptr_tensor_cuda = csr_row_ptr_tensor.cuda()
    csr_col_ptr_tensor_cuda = csr_col_ptr_tensor.cuda()
    edge_weight_tensor_cuda = edge_weight_tensor.cuda()
    feature_tensor_cuda = feature_tensor.cuda()
    output_feature_tensor_cuda = wg_ops.gspmm_weighted_forward(
        csr_row_ptr_tensor_cuda,
        csr_col_ptr_tensor_cuda,
        edge_weight_tensor_cuda,
        feature_tensor_cuda,
    )
    output_feature_tensor = output_feature_tensor_cuda.cpu()
    output_feature_tensor_ref = host_gspmm_weighted_forward(
        csr_row_ptr_tensor, csr_col_ptr_tensor, edge_weight_tensor, feature_tensor
    )
    torch.allclose(output_feature_tensor, output_feature_tensor_ref)


@pytest.mark.parametrize("target_node_count", [101, 113])
@pytest.mark.parametrize("neighbor_node_count", [157, 1987])
@pytest.mark.parametrize("edge_num", [1001, 2302])
@pytest.mark.parametrize("feature_dtype", [torch.float32])
@pytest.mark.parametrize("feature_dim", [32, 128])
@pytest.mark.parametrize("num_head", [8, 32])
def test_gspmm_weighted_forward(
    target_node_count,
    neighbor_node_count,
    edge_num,
    feature_dtype,
    feature_dim,
    num_head,
):
    gpu_count = torch.cuda.device_count()
    assert gpu_count > 0
    routine_func(
        target_node_count=target_node_count,
        neighbor_node_count=neighbor_node_count,
        edge_num=edge_num,
        feature_dtype=feature_dtype,
        feature_dim=feature_dim,
        num_head=num_head,
    )
