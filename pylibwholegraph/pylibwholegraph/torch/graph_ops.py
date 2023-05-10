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
import pylibwholegraph.binding.wholememory_binding as wmb
from .wholegraph_env import (
    get_stream,
    TorchMemoryContext,
    get_wholegraph_env_fns,
    wrap_torch_tensor,
)


def append_unique(
    target_node_tensor: torch.Tensor,
    neighbor_node_tensor: torch.Tensor,
    need_neighbor_raw_to_unique: bool = False,
):
    assert target_node_tensor.dim() == 1
    assert neighbor_node_tensor.dim() == 1
    assert target_node_tensor.is_cuda
    assert neighbor_node_tensor.is_cuda

    output_unique_node_context = TorchMemoryContext()
    output_unique_node_tensor_id = id(output_unique_node_context)
    output_neighbor_raw_to_unique_mapping_tensor = None
    if need_neighbor_raw_to_unique:
        output_neighbor_raw_to_unique_mapping_tensor = torch.empty(
            neighbor_node_tensor.shape[0], device="cuda", dtype=torch.int
        )

    wmb.append_unique(
        wrap_torch_tensor(target_node_tensor),
        wrap_torch_tensor(neighbor_node_tensor),
        output_unique_node_tensor_id,
        wrap_torch_tensor(output_neighbor_raw_to_unique_mapping_tensor),
        get_wholegraph_env_fns(),
        get_stream(),
    )
    if need_neighbor_raw_to_unique:
        return (
            output_unique_node_context.get_tensor(),
            output_neighbor_raw_to_unique_mapping_tensor,
        )
    else:
        return output_unique_node_context.get_tensor()


def spmm_no_weight_forward(
    csr_row_ptr_tensor: torch.Tensor,
    csr_col_ptr_tensor: torch.Tensor,
    feature_tensor: torch.Tensor,
    aggregator: torch.int64,
):
    assert csr_row_ptr_tensor.dim() == 1
    assert csr_col_ptr_tensor.dim() == 1
    assert feature_tensor.dim() == 2
    assert csr_row_ptr_tensor.dtype == torch.int32
    assert csr_col_ptr_tensor.dtype == torch.int32
    assert csr_row_ptr_tensor.is_cuda
    assert csr_col_ptr_tensor.is_cuda
    assert feature_tensor.is_cuda

    output_feature_tensor = torch.empty(
        (csr_row_ptr_tensor.shape[0] - 1, feature_tensor.shape[1]),
        device="cuda",
        dtype=feature_tensor.dtype,
    )

    wmb.spmm_no_weight_forward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(feature_tensor),
        aggregator,
        wrap_torch_tensor(output_feature_tensor),
        get_stream(),
    )

    return output_feature_tensor


def spmm_no_weight_backward(
    csr_row_ptr_tensor: torch.Tensor,
    csr_col_ptr_tensor: torch.Tensor,
    input_grad_feature_tensor: torch.Tensor,
    input_cout: torch.int64,
    aggregator: torch.int64,
):
    assert csr_row_ptr_tensor.dim() == 1
    assert csr_col_ptr_tensor.dim() == 1
    assert csr_row_ptr_tensor.dtype == torch.int32
    assert csr_col_ptr_tensor.dtype == torch.int32
    assert csr_row_ptr_tensor.is_cuda
    assert csr_col_ptr_tensor.is_cuda
    assert input_grad_feature_tensor.is_cuda

    output_grad_feature_tensor = torch.empty(
        (input_cout, input_grad_feature_tensor.shape[1]),
        device="cuda",
        dtype=input_grad_feature_tensor.dtype,
    )

    wmb.spmm_no_weight_backward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(input_grad_feature_tensor),
        aggregator,
        wrap_torch_tensor(output_grad_feature_tensor),
        get_stream(),
    )

    return output_grad_feature_tensor


def spadd_gat_forward(
    csr_row_ptr_tensor: torch.Tensor,
    csr_col_ptr_tensor: torch.Tensor,
    edge_weight_left_tensor: torch.Tensor,
    edge_weight_right_tensor: torch.Tensor,
):
    assert csr_row_ptr_tensor.dim() == 1
    assert csr_col_ptr_tensor.dim() == 1
    assert csr_row_ptr_tensor.dtype == torch.int32
    assert csr_col_ptr_tensor.dtype == torch.int32
    assert edge_weight_left_tensor.dim() == 2
    assert edge_weight_right_tensor.dim() == 2
    assert edge_weight_right_tensor.shape[1] == edge_weight_left_tensor.shape[1]
    assert edge_weight_left_tensor.shape[0] == csr_row_ptr_tensor.shape[0] - 1
    assert csr_row_ptr_tensor.is_cuda
    assert csr_col_ptr_tensor.is_cuda
    assert edge_weight_left_tensor.is_cuda
    assert edge_weight_right_tensor.is_cuda

    output_score_tensor = torch.empty(
        (csr_col_ptr_tensor.shape[0], edge_weight_left_tensor.shape[1]),
        device="cuda",
        dtype=edge_weight_left_tensor.dtype,
    )
    wmb.spadd_gat_forward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(edge_weight_left_tensor),
        wrap_torch_tensor(edge_weight_right_tensor),
        wrap_torch_tensor(output_score_tensor),
        get_stream(),
    )
    return output_score_tensor


def spadd_gat_backward(
    csr_row_ptr_tensor: torch.Tensor,
    csr_col_ptr_tensor: torch.Tensor,
    grad_score_tensor: torch.Tensor,
    neighbor_node_count: torch.int64,
):
    assert csr_row_ptr_tensor.dim() == 1
    assert csr_col_ptr_tensor.dim() == 1
    assert csr_row_ptr_tensor.dtype == torch.int32
    assert csr_col_ptr_tensor.dtype == torch.int32
    assert grad_score_tensor.dim() == 2
    assert grad_score_tensor.shape[0] == csr_col_ptr_tensor.shape[0]
    assert csr_row_ptr_tensor.is_cuda
    assert csr_col_ptr_tensor.is_cuda
    assert grad_score_tensor.is_cuda

    output_edge_weight_left_tensor = torch.empty(
        (csr_row_ptr_tensor.shape[0] - 1, grad_score_tensor.shape[1]),
        device="cuda",
        dtype=grad_score_tensor.dtype,
    )
    output_edge_weight_right_tensor = torch.empty(
        (neighbor_node_count, grad_score_tensor.shape[1]),
        device="cuda",
        dtype=grad_score_tensor.dtype,
    )

    wmb.spadd_gat_backward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(grad_score_tensor),
        wrap_torch_tensor(output_edge_weight_left_tensor),
        wrap_torch_tensor(output_edge_weight_right_tensor),
        get_stream(),
    )

    return output_edge_weight_left_tensor, output_edge_weight_right_tensor


def edge_weight_softmax_forward(
    csr_row_ptr_tensor: torch.Tensor, edge_weight_tensor: torch.Tensor
):
    assert csr_row_ptr_tensor.dim() == 1
    assert edge_weight_tensor.dim() == 2
    assert csr_row_ptr_tensor.is_cuda
    assert edge_weight_tensor.is_cuda

    output_edge_weight_softmax_tensor = torch.empty(
        (edge_weight_tensor.shape[0], edge_weight_tensor.shape[1]),
        device="cuda",
        dtype=edge_weight_tensor.dtype,
    )

    wmb.edge_weight_softmax_forward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(edge_weight_tensor),
        wrap_torch_tensor(output_edge_weight_softmax_tensor),
        get_stream(),
    )

    return output_edge_weight_softmax_tensor


def edge_weight_softmax_backward(
    csr_row_ptr_tensor: torch.Tensor,
    edge_weight_tensor: torch.Tensor,
    grad_edge_weight_softmax_tensor: torch.Tensor,
):
    assert csr_row_ptr_tensor.dim() == 1
    assert edge_weight_tensor.dim() == 2
    assert grad_edge_weight_softmax_tensor.dim() == 2
    assert edge_weight_tensor.dtype == grad_edge_weight_softmax_tensor.dtype
    assert csr_row_ptr_tensor.is_cuda
    assert edge_weight_tensor.is_cuda
    assert grad_edge_weight_softmax_tensor.is_cuda

    output_grad_edge_wieght_tensor = torch.empty(
        (edge_weight_tensor.shape[0], edge_weight_tensor.shape[1]),
        device="cuda",
        dtype=edge_weight_tensor.dtype,
    )
    wmb.edge_weight_softmax_backward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(edge_weight_tensor),
        wrap_torch_tensor(grad_edge_weight_softmax_tensor),
        wrap_torch_tensor(output_grad_edge_wieght_tensor),
        get_stream(),
    )

    return output_grad_edge_wieght_tensor


def add_csr_self_loop(
    csr_row_ptr_tensor: torch.Tensor, csr_col_ptr_tensor: torch.Tensor
):
    assert csr_row_ptr_tensor.dim() == 1
    assert csr_col_ptr_tensor.dim() == 1
    assert csr_row_ptr_tensor.is_cuda
    assert csr_col_ptr_tensor.is_cuda

    output_csr_row_ptr_tensor = torch.empty(
        (csr_row_ptr_tensor.shape[0],), device="cuda", dtype=csr_row_ptr_tensor.dtype
    )
    output_csr_col_ptr_tensor = torch.empty(
        (csr_col_ptr_tensor.shape[0] + csr_row_ptr_tensor.shape[0] - 1,),
        device="cuda",
        dtype=csr_col_ptr_tensor.dtype,
    )
    wmb.add_csr_self_loop(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(output_csr_row_ptr_tensor),
        wrap_torch_tensor(output_csr_col_ptr_tensor),
        get_stream(),
    )
    return output_csr_row_ptr_tensor, output_csr_col_ptr_tensor


def gspmm_weighted_forward(
    csr_row_ptr_tensor: torch.Tensor,
    csr_col_ptr_tensor: torch.Tensor,
    edge_weight_tensor: torch.Tensor,
    feature_tensor: torch.Tensor,
):
    assert csr_row_ptr_tensor.dim() == 1
    assert csr_col_ptr_tensor.dim() == 1
    assert edge_weight_tensor.dim() == 2
    assert feature_tensor.dim() == 3
    assert edge_weight_tensor.shape[1] == feature_tensor.shape[1]
    assert edge_weight_tensor.shape[0] == csr_col_ptr_tensor.shape[0]
    assert edge_weight_tensor.dtype == feature_tensor.dtype
    assert csr_row_ptr_tensor.is_cuda
    assert csr_col_ptr_tensor.is_cuda
    assert edge_weight_tensor.is_cuda
    assert feature_tensor.is_cuda

    output_feature_tensor = torch.empty(
        (
            csr_row_ptr_tensor.shape[0] - 1,
            feature_tensor.shape[1],
            feature_tensor.shape[2],
        ),
        device="cuda",
        dtype=feature_tensor.dtype,
    )
    wmb.gspmm_weighted_forward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(edge_weight_tensor),
        wrap_torch_tensor(feature_tensor),
        wrap_torch_tensor(output_feature_tensor),
        get_stream(),
    )
    return output_feature_tensor


def gspmm_weighted_backward(
    csr_row_ptr_tensor: torch.Tensor,
    csr_col_ptr_tensor: torch.Tensor,
    edge_weight_tensor: torch.Tensor,
    feature_tensor: torch.Tensor,
    grad_feature_tensor: torch.Tensor,
    need_grad_edge_weight: bool = False,
    need_grad_feature: bool = False,
):
    assert csr_row_ptr_tensor.dim() == 1
    assert csr_col_ptr_tensor.dim() == 1
    assert edge_weight_tensor.dim() == 2
    assert feature_tensor.dim() == 3
    assert edge_weight_tensor.shape[1] == feature_tensor.shape[1]
    assert edge_weight_tensor.shape[0] == csr_col_ptr_tensor.shape[0]
    assert edge_weight_tensor.dtype == feature_tensor.dtype
    assert grad_feature_tensor.dtype == feature_tensor.dtype
    assert grad_feature_tensor.shape[0] == csr_row_ptr_tensor.shape[0] - 1
    assert grad_feature_tensor.shape[1] == feature_tensor.shape[1]
    assert grad_feature_tensor.shape[2] == feature_tensor.shape[2]
    assert csr_row_ptr_tensor.is_cuda
    assert csr_col_ptr_tensor.is_cuda
    assert edge_weight_tensor.is_cuda
    assert feature_tensor.is_cuda
    assert grad_feature_tensor.is_cuda

    output_grad_edge_weight_tensor = None
    output_grad_feature_tensor = None
    if need_grad_edge_weight:
        output_grad_edge_weight_tensor = torch.empty(
            (edge_weight_tensor.shape[0], edge_weight_tensor.shape[1]),
            device="cuda",
            dtype=edge_weight_tensor.dtype,
        )

    if need_grad_feature:
        output_grad_feature_tensor = torch.empty(
            (feature_tensor.shape[0], feature_tensor.shape[1], feature_tensor.shape[2]),
            device="cuda",
            dtype=feature_tensor.dtype,
        )

    wmb.gspmm_weighted_backward(
        wrap_torch_tensor(csr_row_ptr_tensor),
        wrap_torch_tensor(csr_col_ptr_tensor),
        wrap_torch_tensor(edge_weight_tensor),
        wrap_torch_tensor(feature_tensor),
        wrap_torch_tensor(grad_feature_tensor),
        wrap_torch_tensor(output_grad_edge_weight_tensor),
        wrap_torch_tensor(output_grad_feature_tensor),
        get_stream(),
    )
    if need_grad_edge_weight and need_grad_feature:
        return output_grad_edge_weight_tensor, output_grad_feature_tensor
    elif need_grad_feature:
        return output_grad_feature_tensor
    elif need_grad_edge_weight:
        return output_grad_edge_weight_tensor
    else:
        return None
