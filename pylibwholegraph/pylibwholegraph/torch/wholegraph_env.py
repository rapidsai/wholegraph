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
from typing import Union
from .utils import wholememory_dtype_to_torch_dtype, torch_dtype_to_wholememory_dtype


default_cuda_stream_int_ptr = None
default_wholegraph_env_context = None


def get_stream(use_default=True):
    global default_cuda_stream_int_ptr
    cuda_stream_int_ptr = None
    if default_cuda_stream_int_ptr is None or not use_default:
        cuda_stream = torch.cuda.current_stream()._as_parameter_
        if cuda_stream.value is not None:
            cuda_stream_int_ptr = cuda_stream.value
        else:
            cuda_stream_int_ptr = int(0)
        if use_default:
            default_cuda_stream_int_ptr = cuda_stream_int_ptr
    else:
        cuda_stream_int_ptr = default_cuda_stream_int_ptr
    return cuda_stream_int_ptr


class TorchEmptyGlobalContext(object):
    def __init__(self):
        pass


class TorchMemoryContext(object):
    def __init__(self):
        self.tensor = None

    def set_tensor(self, t: torch.Tensor):
        self.tensor = t

    def get_tensor(self):
        return self.tensor

    def free(self):
        self.tensor = None


def torch_create_memory_context_env_fn(
    global_context: TorchEmptyGlobalContext,
) -> TorchMemoryContext:
    t = TorchMemoryContext()
    # print('torch_create_memory_context_env_fn t=%d' % (id(t), ))
    return t


def torch_destroy_memory_context_env_fn(
    memory_context: TorchMemoryContext, global_context: TorchEmptyGlobalContext
):
    pass


def torch_malloc_env_fn(
    tensor_desc: wmb.PyWholeMemoryTensorDescription,
    malloc_type: wmb.PyMemoryAllocType,
    memory_context: TorchMemoryContext,
    global_context: TorchEmptyGlobalContext,
) -> int:
    # print('already in torch_malloc_env_fn', file=sys.stderr)
    pinned = False
    device = None
    # print('torch_malloc_env_fn before config, type=%d' % (malloc_type.get_type(), ), file=sys.stderr)
    if malloc_type.get_type() == wmb.WholeMemoryMemoryAllocType.MatDevice:
        device = torch.device("cuda")
    elif malloc_type.get_type() == wmb.WholeMemoryMemoryAllocType.MatHost:
        device = torch.device("cpu")
    else:
        assert malloc_type.get_type() == wmb.WholeMemoryMemoryAllocType.MatPinned
        device = torch.device("cpu")
        pinned = True
    # print('torch_malloc_env_fn after config', file=sys.stderr)
    shape = tensor_desc.shape
    # print('torch_malloc_env_fn after shape', file=sys.stderr)
    dtype = wholememory_dtype_to_torch_dtype(tensor_desc.dtype)
    # print('torch_malloc_env_fn after dtype', file=sys.stderr)
    t = torch.empty(shape, dtype=dtype, device=device, pin_memory=pinned)
    memory_context.set_tensor(t)
    # print('torch_malloc_env_fn done return=%ld' % (t.data_ptr(), ), file=sys.stderr)
    return t.data_ptr()


def torch_free_env_fn(
    memory_context: TorchMemoryContext, global_context: TorchEmptyGlobalContext
):
    memory_context.free()


def create_current_env_context():
    # print('in wholegraph_env.py create_current_env_context')
    context = wmb.GlobalContextWrapper()
    global_context = TorchEmptyGlobalContext()
    context.create_context(
        torch_create_memory_context_env_fn,
        torch_destroy_memory_context_env_fn,
        torch_malloc_env_fn,
        torch_free_env_fn,
        global_context,
        torch_malloc_env_fn,
        torch_free_env_fn,
        global_context,
    )
    return context


def get_wholegraph_env_fns(use_default=True) -> int:
    global default_wholegraph_env_context
    wholegraph_env_context = None
    if default_wholegraph_env_context is None or not use_default:
        wholegraph_env_context = create_current_env_context()
        if use_default:
            default_wholegraph_env_context = wholegraph_env_context
    else:
        wholegraph_env_context = default_wholegraph_env_context
    return wholegraph_env_context.get_env_fns()


def wrap_torch_tensor(t: Union[torch.Tensor, None]) -> wmb.WrappedLocalTensor:
    py_desc = wmb.PyWholeMemoryTensorDescription()
    wm_t = wmb.WrappedLocalTensor()
    if t is None:
        return wm_t.wrap_tensor(py_desc, 0)
    py_desc.set_dtype(torch_dtype_to_wholememory_dtype(t.dtype))
    py_desc.set_storage_offset(0)
    py_desc.set_shape(tuple(t.shape))
    py_desc.set_stride(tuple(t.stride()))
    return wm_t.wrap_tensor(py_desc, t.data_ptr())
