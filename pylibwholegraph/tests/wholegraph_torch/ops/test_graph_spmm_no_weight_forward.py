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


def host_general_spmm_forward(
    csr_row_ptr_tensor, csr_col_ptr_tensor, feature_tensor, aggregator
):
    row_num = csr_row_ptr_tensor.shape[0] - 1
    feature_dim = feature_tensor.shape[1]
    output_tensor = torch.zeros((row_num, feature_dim), dtype=feature_tensor.dtype)

    for i in range(row_num):
        start = csr_row_ptr_tensor[i]
        end = csr_row_ptr_tensor[i + 1]
        count = end - start
        mean = 1.0
        if count > 0:
            if aggregator == 1:
                mean /= float(count)
            if aggregator == 2:
                mean /= float(count + 1)
        for k in range(feature_dim):
            agg_value = 0.0
            if aggregator == 2:
                agg_value += feature_tensor[i][k]
            for j in range(start, end):
                col_id = csr_col_ptr_tensor[j]
                agg_value += feature_tensor[col_id][k]
            agg_value *= mean
            output_tensor[i][k] = agg_value
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
    feature_tensor = torch.rand((neighbor_node_count, feature_dim), dtype=feature_dtype)
    csr_row_ptr_tensor_cuda = csr_row_ptr_tensor.cuda()
    csr_col_ptr_tensor_cuda = csr_col_ptr_tensor.cuda()
    feature_tensor_cuda = feature_tensor.cuda()
    output_feature_tensor_cuda = wg_ops.spmm_no_weight_forward(
        csr_row_ptr_tensor=csr_row_ptr_tensor_cuda,
        csr_col_ptr_tensor=csr_col_ptr_tensor_cuda,
        feature_tensor=feature_tensor_cuda,
        aggregator=aggregator,
    )
    output_feature_tensor = output_feature_tensor_cuda.cpu()

    output_feature_tensor_ref = host_general_spmm_forward(
        csr_row_ptr_tensor, csr_col_ptr_tensor, feature_tensor, aggregator
    )

    assert torch.allclose(output_feature_tensor, output_feature_tensor_ref, 1e-03)


@pytest.mark.parametrize("target_node_count", [101, 113])
@pytest.mark.parametrize("neighbor_node_count", [157, 1987])
@pytest.mark.parametrize("edge_num", [1001, 2302])
@pytest.mark.parametrize("feature_dim", [128])
@pytest.mark.parametrize("feature_dtype", [torch.float32])
@pytest.mark.parametrize("aggregator", [0, 1, 2])  # 0: sum, 1: mean, 2: gcn_mean
def test_spmm_no_weight_forward(
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
