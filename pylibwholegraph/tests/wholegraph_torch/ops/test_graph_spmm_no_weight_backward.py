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


def host_general_spmm_backward(
    csr_row_ptr_tensor,
    csr_col_ptr_tensor,
    input_grad_feature_tensor,
    output_count,
    aggregator,
):
    row_num = csr_row_ptr_tensor.shape[0] - 1
    feature_dim = input_grad_feature_tensor.shape[1]
    assert row_num == input_grad_feature_tensor.shape[0]
    output_tensor = torch.zeros(
        (output_count, feature_dim), dtype=input_grad_feature_tensor.dtype
    )

    for i in range(row_num):
        start = csr_row_ptr_tensor[i]
        end = csr_row_ptr_tensor[i + 1]
        count = end - start
        for k in range(feature_dim):
            value = input_grad_feature_tensor[i][k]
            if aggregator == 1:
                if count > 0:
                    value /= float(count)
            if aggregator == 2:
                value /= float(count + 1)
                output_tensor[i][k] += value
            for j in range(start, end):
                col_id = csr_col_ptr_tensor[j]
                output_tensor[col_id][k] += value
    return output_tensor


def routine_func(**kwargs):
    load_wholegraph_op_libraries()
    target_node_count = kwargs["target_node_count"]
    neighbor_node_count = kwargs["neighbor_node_count"]
    edge_num = kwargs["edge_num"]
    feature_dim = kwargs["feature_dim"]
    feature_dtype = kwargs["feature_dtype"]
    aggregator = kwargs["aggregator"]
    assert neighbor_node_count >= target_node_count
    csr_row_ptr_tensor, csr_col_ptr_tensor, _ = gen_csr_graph(
        target_node_count,
        edge_num,
        neighbor_node_count,
        csr_row_dtype=torch.int32,
        csr_col_dtype=torch.int32,
    )
    input_grad_feature_tensor = torch.rand(
        (target_node_count, feature_dim), dtype=feature_dtype
    )
    csr_row_ptr_tensor_cuda = csr_row_ptr_tensor.cuda()
    csr_col_ptr_tensor_cuda = csr_col_ptr_tensor.cuda()
    input_grad_feature_tensor_cuda = input_grad_feature_tensor.cuda()
    output_grad_feature_tensor_cuda = wg_ops.spmm_no_weight_backward(
        csr_row_ptr_tensor=csr_row_ptr_tensor_cuda,
        csr_col_ptr_tensor=csr_col_ptr_tensor_cuda,
        input_grad_feature_tensor=input_grad_feature_tensor_cuda,
        input_cout=neighbor_node_count,
        aggregator=aggregator,
    )
    output_grad_feature_tensor = output_grad_feature_tensor_cuda.cpu()

    output_grad_feature_tensor_ref = host_general_spmm_backward(
        csr_row_ptr_tensor,
        csr_col_ptr_tensor,
        input_grad_feature_tensor,
        neighbor_node_count,
        aggregator,
    )

    assert torch.allclose(
        output_grad_feature_tensor, output_grad_feature_tensor_ref, 1e-03
    )


@pytest.mark.parametrize("target_node_count", [101, 113])
@pytest.mark.parametrize("neighbor_node_count", [189, 1987])
@pytest.mark.parametrize("edge_num", [1001, 2302])
@pytest.mark.parametrize("feature_dim", [128])
@pytest.mark.parametrize("feature_dtype", [torch.float32])
@pytest.mark.parametrize("aggregator", [0, 1, 2])  # 0: sum, 1: mean, 2: gcn_mean
def test_spmm_no_weight_backward(
    target_node_count,
    neighbor_node_count,
    edge_num,
    feature_dim,
    feature_dtype,
    aggregator,
):
    gpu_count = torch.cuda.device_count()
    assert gpu_count > 0
    routine_func(
        target_node_count=target_node_count,
        neighbor_node_count=neighbor_node_count,
        edge_num=edge_num,
        feature_dim=feature_dim,
        feature_dtype=feature_dtype,
        aggregator=aggregator,
    )
