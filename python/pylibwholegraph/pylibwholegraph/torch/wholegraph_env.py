# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
import os.path
import importlib

import torch
import pylibwholegraph
import pylibwholegraph.binding.wholememory_binding as wmb
from typing import Union
from .utils import wholememory_dtype_to_torch_dtype, torch_dtype_to_wholememory_dtype

default_wholegraph_env_context = None
torch_cpp_ext_loaded = False
torch_cpp_ext_lib = None


def get_stream():
    cuda_stream_int_ptr = None
    cuda_stream = torch.cuda.current_stream()._as_parameter_
    if cuda_stream.value is not None:
        cuda_stream_int_ptr = cuda_stream.value
    else:
        cuda_stream_int_ptr = int(0)
    return cuda_stream_int_ptr


class TorchEmptyGlobalContext(object):
    def __init__(self):
        pass


class TorchMemoryContext(object):
    def __init__(self):
        self.tensor = None
        if torch_cpp_ext_loaded:
            self.handle = torch_cpp_ext_lib.create_output_context()
        else:
            self.handle = 0

    def __del__(self):
        self.free()

    def get_c_context(self):
        if torch_cpp_ext_loaded:
            return self.handle
        else:
            return id(self)

    def set_tensor(self, t: torch.Tensor):
        self.tensor = t

    def get_handle(self):
        return self.handle

    def get_tensor(self):
        if torch_cpp_ext_loaded:
            self.tensor = torch_cpp_ext_lib.get_tensor_from_context(self.handle)
            return self.tensor
        else:
            return self.tensor

    def free(self):
        self.tensor = None
        if torch_cpp_ext_loaded and self.get_handle() != 0:
            torch_cpp_ext_lib.destroy_output_context(self.get_handle())
            self.handle = 0

    def free_data(self):
        self.tensor = None
        if torch_cpp_ext_loaded and self.get_handle() != 0:
            torch_cpp_ext_lib.free_context_data(self.get_handle())


def torch_create_memory_context_env_fn(
    global_context: TorchEmptyGlobalContext,
) -> TorchMemoryContext:
    t = TorchMemoryContext()
    return t


def torch_destroy_memory_context_env_fn(
    memory_context: TorchMemoryContext, global_context: TorchEmptyGlobalContext
):
    memory_context.free()


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
    memory_context.free_data()


class ExtContextWrapper(object):
    def __init__(self, env_func: int):
        self.env_func = env_func

    def get_env_fns(self) -> int:
        return self.env_func


def create_current_env_context():
    # print('in wholegraph_env.py create_current_env_context')
    global torch_cpp_ext_loaded
    global torch_cpp_ext_lib
    if torch_cpp_ext_loaded:
        return ExtContextWrapper(torch_cpp_ext_lib.get_wholegraph_env_fns())
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


def get_cpp_extension_src_path():
    return os.path.dirname(pylibwholegraph.__file__)


def compile_cpp_extension():
    import torch.utils.cpp_extension

    global torch_cpp_ext_loaded
    global torch_cpp_ext_lib
    cpp_extension_path = os.path.join(get_cpp_extension_src_path(), "torch_cpp_ext")
    extra_cflags = []
    extra_ldflags = ["-lwholegraph"]
    if "CONDA_PREFIX" in os.environ:
        extra_cflags.append(
            "".join(["-I", os.path.join(os.environ["CONDA_PREFIX"], "include")])
        )
        extra_ldflags.append(
            "".join(["-L", os.path.join(os.environ["CONDA_PREFIX"], "lib")])
        )
    if "LIBWHOLEGRAPH_DIR" in os.environ:
        extra_cflags.append(
            "".join(["-I", os.path.join(os.environ["LIBWHOLEGRAPH_DIR"], "include")])
        )
        extra_ldflags.append(
            "".join(["-L", os.path.join(os.environ["LIBWHOLEGRAPH_DIR"], "lib")])
        )
    torch.utils.cpp_extension.load(
        name="pylibwholegraph.pylibwholegraph_torch_ext",
        sources=[
            os.path.join(cpp_extension_path, "wholegraph_torch_ext.cpp"),
            os.path.join(cpp_extension_path, "torch_env_func_ptrs.cpp"),
            os.path.join(cpp_extension_path, "torch_utils.cpp"),
        ],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        with_cuda=True,
        verbose=True,
    )
    torch_cpp_ext_lib = importlib.import_module(
        "pylibwholegraph.pylibwholegraph_torch_ext"
    )
    torch_cpp_ext_loaded = True
