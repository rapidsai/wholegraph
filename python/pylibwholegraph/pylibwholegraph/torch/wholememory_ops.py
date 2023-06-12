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
    get_wholegraph_env_fns,
    wrap_torch_tensor,
)
from .utils import wholememory_dtype_to_torch_dtype


def wholememory_gather_forward_functor(
    wholememory_tensor: wmb.PyWholeMemoryTensor,
    indices_tensor: torch.Tensor,
    requires_grad=False,
    torch_output_dtype=None,
):
    """
    Wrapper functor for gather op of WholeMemory Tensor
    :param wholememory_tensor: PyWholeMemoryTensor
    :param indices_tensor: Indices to gather from
    :param requires_grad: if requires gradients
    :param torch_output_dtype: output dtype, None for same as wholememory_tensor
    :return: Gathered tensor
    """
    assert indices_tensor.dim() == 1
    assert indices_tensor.dtype == torch.int32 or indices_tensor.dtype == torch.int64
    if torch_output_dtype is None:
        torch_output_dtype = wholememory_dtype_to_torch_dtype(wholememory_tensor.dtype)
    output_tensor = torch.empty(
        [indices_tensor.shape[0], wholememory_tensor.shape[1]],
        device="cuda",
        dtype=torch_output_dtype,
        requires_grad=requires_grad,
    )
    wmb.wholememory_gather_op(
        wholememory_tensor,
        wrap_torch_tensor(indices_tensor),
        wrap_torch_tensor(output_tensor),
        get_wholegraph_env_fns(),
        get_stream(),
    )
    return output_tensor


def wholememory_scatter_functor(
    input_tensor: torch.Tensor,
    indices_tensor: torch.Tensor,
    wholememory_tensor: wmb.PyWholeMemoryTensor,
):
    """
    Wrapper functor for scatter op of WholeMemory Tensor
    :param input_tensor: Input tensor to scater to WholeMemory Tensor
    :param indices_tensor: Indices to scatter to
    :param wholememory_tensor: WholeMemory Tensor
    :return: None
    """
    assert indices_tensor.dim() == 1
    assert indices_tensor.dtype == torch.int32 or indices_tensor.dtype == torch.int64
    wmb.wholememory_scatter_op(
        wrap_torch_tensor(input_tensor),
        wrap_torch_tensor(indices_tensor),
        wholememory_tensor,
        get_wholegraph_env_fns(),
        get_stream(),
    )
