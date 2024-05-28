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

import pylibwholegraph.binding.wholememory_binding as wmb
import torch
from .utils import (
    torch_dtype_to_wholememory_dtype,
    wholememory_dtype_to_torch_dtype,
    get_file_size,
)
from .utils import str_to_wmb_wholememory_memory_type, str_to_wmb_wholememory_location
from .utils import get_part_file_name, get_part_file_list
from .comm import WholeMemoryCommunicator
from typing import Union, List
from .dlpack_utils import torch_import_from_dlpack
from .wholegraph_env import wrap_torch_tensor, get_wholegraph_env_fns, get_stream


WholeMemoryMemoryType = wmb.WholeMemoryMemoryType
WholeMemoryMemoryLocation = wmb.WholeMemoryMemoryLocation


class WholeMemoryTensor(object):
    r"""WholeMemory Tensor"""

    def __init__(self, wmb_tensor: wmb.PyWholeMemoryTensor):
        self.wmb_tensor = wmb_tensor

    @property
    def dtype(self):
        return wholememory_dtype_to_torch_dtype(self.wmb_tensor.dtype)

    def dim(self):
        return self.wmb_tensor.dim()

    @property
    def shape(self):
        return self.wmb_tensor.shape

    def stride(self):
        return self.wmb_tensor.stride()

    def storage_offset(self):
        return self.wmb_tensor.storage_offset()

    def get_comm(self):
        return WholeMemoryCommunicator(
            self.wmb_tensor.get_wholememory_handle().get_communicator()
        )

    def gather(self,
               indice: torch.Tensor,
               *,
               force_dtype: Union[torch.dtype, None] = None):
        assert indice.dim() == 1
        embedding_dim = self.shape[1]
        embedding_count = indice.shape[0]
        current_cuda_device = "cuda:%d" % (torch.cuda.current_device(),)
        output_dtype = (
            force_dtype if force_dtype is not None else self.dtype
        )
        output_tensor = torch.empty(
            [embedding_count, embedding_dim],
            device=current_cuda_device,
            dtype=output_dtype,
            requires_grad=False,
        )
        wmb.wholememory_gather_op(self.wmb_tensor,
                                  wrap_torch_tensor(indice),
                                  wrap_torch_tensor(output_tensor),
                                  get_wholegraph_env_fns(),
                                  get_stream())
        return output_tensor

    def scatter(self,
                input_tensor: torch.Tensor,
                indice: torch.Tensor):
        assert indice.dim() == 1
        assert input_tensor.dim() == 2
        assert indice.shape[0] == input_tensor.shape[0]
        assert input_tensor.shape[1] == self.shape[1]
        wmb.wholememory_scatter_op(wrap_torch_tensor(input_tensor),
                                   wrap_torch_tensor(indice),
                                   self.wmb_tensor,
                                   get_wholegraph_env_fns(),
                                   get_stream())

    def get_sub_tensor(self, starts, ends):
        """
        Get sub tensor of WholeMemory Tensor
        :param starts: An array of the start indices of each dim
        :param ends: An array of the end indices of each dim, -1 means to the last element
        :return: WholeMemory Tensor
        """
        return WholeMemoryTensor(self.wmb_tensor.get_sub_tensor(starts, ends))

    def get_local_tensor(self, host_view: bool = False):
        """Get local tensor of WholeMemory Tensor
        :param host_view: Get host view or not, if True, return host tensor, else return device tensor
        :return: Tuple of DLPack Tensor and element offset.
        """
        if host_view:
            return self.wmb_tensor.get_local_tensor(
                torch_import_from_dlpack, WholeMemoryMemoryLocation.MlHost, -1
            )
        else:
            return self.wmb_tensor.get_local_tensor(
                torch_import_from_dlpack,
                WholeMemoryMemoryLocation.MlDevice,
                torch.cuda.current_device(),
            )

    def get_global_tensor(self, host_view: bool = False):
        """Get global tensor of WholeMemory Tensor
        :param host_view: Get host view or not, if True, return host tensor, else return device tensor
        :return: Tuple of DLPack Tensor and element offset (0 for global tensor).
        """
        if host_view:
            return self.wmb_tensor.get_global_tensor(
                torch_import_from_dlpack, WholeMemoryMemoryLocation.MlHost, -1
            )
        else:
            return self.wmb_tensor.get_global_tensor(
                torch_import_from_dlpack,
                WholeMemoryMemoryLocation.MlDevice,
                torch.cuda.current_device(),
            )

    def get_all_chunked_tensor(self, host_view: bool = False):
        """Get all chunked tensor of WholeMemory Tensor
        :param host_view: Get host view or not, if True, return host tensor, else return device tensor
        :return: Tuple of DLPack Tensors and element offsets.
        """
        if host_view:
            return self.wmb_tensor.get_global_tensorget_all_chunked_tensor(
                torch_import_from_dlpack, WholeMemoryMemoryLocation.MlHost, -1
            )
        else:
            return self.wmb_tensor.get_global_tensorget_all_chunked_tensor(
                torch_import_from_dlpack,
                WholeMemoryMemoryLocation.MlDevice,
                torch.cuda.current_device(),
            )

    def from_filelist(self, filelist: Union[List[str], str], round_robin_size: int = 0):
        """
        Load WholeMemory Tensor from file lists
        :param filelist: file list to load from
        :param round_robin_size: continuous embedding size of a rank using round robin shard strategy
        :return: None
        """
        if isinstance(filelist, str):
            filelist = [filelist]
        self.wmb_tensor.from_filelist(filelist, round_robin_size)

    def from_file_prefix(self, file_prefix: str, part_count: Union[int, None] = None):
        """
        Load WholeMemory  tensor from files with same prefix, files has format
            "%s_part_%d_of_%d" % (prefix, part_id, part_count)
        :param file_prefix: file name prefix
        :param part_count: part count of file
        :return: None
        """
        if part_count is None:
            part_count = self.get_comm().get_size()
        file_list = get_part_file_list(file_prefix, part_count)
        self.from_filelist(file_list)

    def local_to_file(self, filename: str):
        """
        Store local tensor of WholeMemory Tensor to file, all ranks should call this together with different filename
        :param filename: file name of local tensor file.
        :return: None
        """
        self.wmb_tensor.to_file(filename)

    def to_file_prefix(self, file_prefix: str):
        """
        Store WholeMemory Tensor to files with same prefix.
        :param file_prefix: file name prefix
        :return: None
        """
        wm_comm = self.get_comm()
        filename = get_part_file_name(
            file_prefix, wm_comm.get_rank(), wm_comm.get_size()
        )
        self.local_to_file(filename)


def create_wholememory_tensor(
    comm: WholeMemoryCommunicator,
    memory_type: str,
    memory_location: str,
    sizes: List[int],
    dtype: torch.dtype,
    strides: List[int],
):
    """
    Create empty WholeMemory Tensor. Now only support dim = 1 or 2
    :param comm: WholeMemoryCommunicator
    :param memory_type: WholeMemory type, should be continuous, chunked or distributed
    :param memory_location: WholeMemory location, should be cpu or cuda
    :param sizes: size of the tensor
    :param dtype: data type of the tensor
    :param strides: strides of the tensor
    :return: Allocated WholeMemoryTensor
    """
    dim = len(sizes)
    if dim < 1 or dim > 2:
        raise ValueError("Only dim 1 or 2 is supported now.")
    if strides is None:
        strides = [1] * dim
        strides[0] = sizes[1] if dim == 2 else 1
    else:
        assert len(strides) == dim
        assert strides[-1] == 1
        if dim == 2:
            assert strides[0] >= sizes[1]
    td = wmb.PyWholeMemoryTensorDescription()
    td.set_shape(sizes)
    td.set_stride(strides)
    td.set_dtype(torch_dtype_to_wholememory_dtype(dtype))

    wm_memory_type = str_to_wmb_wholememory_memory_type(memory_type)
    wm_location = str_to_wmb_wholememory_location(memory_location)

    return WholeMemoryTensor(
        wmb.create_wholememory_tensor(td, comm.wmb_comm, wm_memory_type, wm_location)
    )


def create_wholememory_tensor_from_filelist(
    comm: WholeMemoryCommunicator,
    memory_type: str,
    memory_location: str,
    filelist: Union[List[str], str],
    dtype: torch.dtype,
    last_dim_size: int = 0,
    last_dim_strides: int = -1,
):
    """
    Create WholeMemory Tensor from list of binary files.
    :param comm: WholeMemoryCommunicator
    :param memory_type: WholeMemory type, should be continuous, chunked or distributed
    :param memory_location: WholeMemory location, should be cpu or cuda
    :param filelist: list of binary files
    :param dtype: data type of the tensor
    :param last_dim_size: 0 for create 1-D array, positive value for create matrix column size
    :param last_dim_strides: stride of last_dim, -1 for same as size of last dim.
    :return: WholeMemoryTensor
    """
    if isinstance(filelist, str):
        filelist = [filelist]
    element_size = torch.tensor([], dtype=dtype).element_size()
    if last_dim_strides == -1:
        last_dim_strides = last_dim_size if last_dim_size > 0 else 1
    file_entry_size = (
        element_size * last_dim_size if last_dim_size > 0 else element_size
    )
    total_file_size = 0
    for filename in filelist:
        file_size = get_file_size(filename)
        if file_size % file_entry_size != 0:
            raise ValueError(
                "File %s size is %d not mutlple of %d"
                % (filename, file_size, file_entry_size)
            )
        total_file_size += file_size
    total_entry_count = total_file_size // file_entry_size
    if last_dim_size == 0:
        sizes = [total_entry_count]
        strides = [1]
    else:
        sizes = [total_entry_count, last_dim_size]
        strides = [last_dim_strides, 1]
    wm_tensor = create_wholememory_tensor(
        comm, memory_type, memory_location, sizes, dtype, strides
    )
    wm_tensor.from_filelist(filelist)
    return wm_tensor


def destroy_wholememory_tensor(wm_tensor: WholeMemoryTensor):
    """
    Destroy allocated WholeMemory Tensor
    :param wm_tensor: WholeMemory Tensor
    :return: None
    """
    wmb.destroy_wholememory_tensor(wm_tensor.wmb_tensor)
    wm_tensor.wmb_tensor = None
