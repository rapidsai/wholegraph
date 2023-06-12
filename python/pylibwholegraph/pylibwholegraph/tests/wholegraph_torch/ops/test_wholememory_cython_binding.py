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
import pylibwholegraph.binding.wholememory_binding as wmb
import torch
from pylibwholegraph.torch.wholegraph_env import (
    get_stream,
    get_wholegraph_env_fns,
    wrap_torch_tensor,
    TorchMemoryContext,
)

import time


def test_smoke():
    torch.cuda.set_device(0)
    output_len = 128
    embed_dim = 10
    input_tensor = torch.ones((embed_dim,), device="cuda")
    indice_tensor = torch.arange(output_len, device="cuda")
    ref_tensor = input_tensor.expand((output_len, embed_dim)) + indice_tensor.reshape(
        (output_len, 1)
    ).expand((output_len, embed_dim))
    output_tensor = torch.empty((output_len, embed_dim), device="cuda")
    assert wmb.py_get_wholememory_tensor_count() == 0

    output_device_context = TorchMemoryContext()
    output_pinned_context = TorchMemoryContext()
    output_host_context = TorchMemoryContext()
    wrapped_input = wrap_torch_tensor(input_tensor)
    wrapped_output = wrap_torch_tensor(output_tensor)

    assert wmb.py_get_wholememory_tensor_count() > 0
    env_func_int_ptr = get_wholegraph_env_fns()
    stream_int_ptr = get_stream()
    wmb.wholememory_env_test_cython_op(
        wrapped_input,
        wrapped_output,
        output_device_context.get_c_context(),
        output_pinned_context.get_c_context(),
        output_host_context.get_c_context(),
        output_len,
        env_func_int_ptr,
        stream_int_ptr,
    )
    torch.cuda.synchronize()
    assert torch.allclose(ref_tensor, output_device_context.get_tensor().cuda())
    assert torch.allclose(ref_tensor, output_pinned_context.get_tensor().cuda())
    assert torch.allclose(ref_tensor, output_host_context.get_tensor().cuda())

    del wrapped_input, wrapped_output

    assert wmb.py_get_wholememory_tensor_count() == 0


def test_loop_memory():
    torch.cuda.set_device(0)
    embedding_dim = 1
    output_len = 1
    input_tensor = torch.ones((embedding_dim,), device="cuda")
    output_tensor = torch.empty((output_len, embedding_dim), device="cuda")
    env_func_int_ptr = get_wholegraph_env_fns()
    stream_int_ptr = get_stream()
    output_device_context = TorchMemoryContext()
    output_pinned_context = TorchMemoryContext()
    output_host_context = TorchMemoryContext()
    wrapped_input = wrap_torch_tensor(input_tensor)
    wrapped_output = wrap_torch_tensor(output_tensor)
    wmb.wholememory_env_test_cython_op(
        wrapped_input,
        wrapped_output,
        output_device_context.get_c_context(),
        output_pinned_context.get_c_context(),
        output_host_context.get_c_context(),
        output_len,
        env_func_int_ptr,
        stream_int_ptr,
    )
    del wrapped_input, wrapped_output
    torch.cuda.synchronize()

    start_time = time.time()
    for i in range(100000):
        output_device_context = TorchMemoryContext()
        output_pinned_context = TorchMemoryContext()
        output_host_context = TorchMemoryContext()
        wrapped_input = wrap_torch_tensor(input_tensor)
        wrapped_output = wrap_torch_tensor(output_tensor)
        wmb.wholememory_env_test_cython_op(
            wrapped_input,
            wrapped_output,
            output_device_context.get_c_context(),
            0,
            0,
            output_len,
            env_func_int_ptr,
            stream_int_ptr,
        )
    del wrapped_input, wrapped_output
    torch.cuda.synchronize()
    end_time = time.time()
    assert wmb.py_get_wholememory_tensor_count() == 0
    print("total_time=%f" % (end_time - start_time,))


@pytest.mark.parametrize("output_len", list(range(1, 100, 17)))
@pytest.mark.parametrize("embed_dim", list(range(1, 128, 23)))
def test_random_alloc(output_len, embed_dim):
    torch.cuda.set_device(0)
    input_tensor = torch.rand((embed_dim,), device="cuda")
    indice_tensor = torch.arange(output_len, device="cuda")
    ref_tensor = input_tensor.expand((output_len, embed_dim)) + indice_tensor.reshape(
        (output_len, 1)
    ).expand((output_len, embed_dim))
    output_tensor = torch.empty((output_len, embed_dim), device="cuda")
    output_device_context = TorchMemoryContext()
    output_pinned_context = TorchMemoryContext()
    output_host_context = TorchMemoryContext()
    wrapped_input = wrap_torch_tensor(input_tensor)
    wrapped_output = wrap_torch_tensor(output_tensor)
    env_func_int_ptr = get_wholegraph_env_fns()
    stream_int_ptr = get_stream()

    wmb.wholememory_env_test_cython_op(
        wrapped_input,
        wrapped_output,
        output_device_context.get_c_context(),
        output_pinned_context.get_c_context(),
        output_host_context.get_c_context(),
        output_len,
        env_func_int_ptr,
        stream_int_ptr,
    )
    torch.cuda.synchronize()
    assert torch.allclose(ref_tensor, output_device_context.get_tensor().cuda())
    assert torch.allclose(ref_tensor, output_pinned_context.get_tensor().cuda())
    assert torch.allclose(ref_tensor, output_host_context.get_tensor().cuda())
