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
