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
import os


WholeMemoryDataType = wmb.WholeMemoryDataType


def torch_dtype_to_wholememory_dtype(torch_dtype: torch.dtype):
    """
    Convert torch.dtype to WholeMemoryDataType
    :param torch_dtype: torch.dtype
    :return: WholeMemoryDataType
    """
    if torch_dtype == torch.float:
        return WholeMemoryDataType.DtFloat
    elif torch_dtype == torch.half:
        return WholeMemoryDataType.DtHalf
    elif torch_dtype == torch.double:
        return WholeMemoryDataType.DtDouble
    elif torch_dtype == torch.bfloat16:
        return WholeMemoryDataType.DtBF16
    elif torch_dtype == torch.int:
        return WholeMemoryDataType.DtInt
    elif torch_dtype == torch.int64:
        return WholeMemoryDataType.DtInt64
    elif torch_dtype == torch.int16:
        return WholeMemoryDataType.DtInt16
    elif torch_dtype == torch.int8:
        return WholeMemoryDataType.DtInt8
    else:
        raise ValueError("torch_dtype: %s not supported" % (torch_dtype,))


def wholememory_dtype_to_torch_dtype(wm_dtype: WholeMemoryDataType):
    """
    Convert WholeMemoryDataType to torch.dtype
    :param wm_dtype: WholeMemoryDataType
    :return: torch.dtype
    """
    if wm_dtype == WholeMemoryDataType.DtFloat:
        return torch.float
    elif wm_dtype == WholeMemoryDataType.DtHalf:
        return torch.half
    elif wm_dtype == WholeMemoryDataType.DtDouble:
        return torch.double
    elif wm_dtype == WholeMemoryDataType.DtBF16:
        return torch.bfloat16
    elif wm_dtype == WholeMemoryDataType.DtInt:
        return torch.int
    elif wm_dtype == WholeMemoryDataType.DtInt64:
        return torch.int64
    elif wm_dtype == WholeMemoryDataType.DtInt16:
        return torch.int16
    elif wm_dtype == WholeMemoryDataType.DtInt8:
        return torch.int8
    else:
        raise ValueError("WholeMemoryMemory: %s not supported" % (int(wm_dtype),))


def get_file_size(filename: str):
    """
    Get file size.
    :param filename: file name
    :return: size of file
    """
    if not os.path.isfile(filename):
        raise ValueError("File %s not found or not file" % (filename,))
    if not os.access(filename, os.R_OK):
        raise ValueError("File %s not readable" % (filename,))
    file_size = os.path.getsize(filename)
    return file_size


def str_to_wmb_wholememory_memory_type(str_wmb_type: str):
    if str_wmb_type == "continuous":
        return wmb.WholeMemoryMemoryType.MtContinuous
    elif str_wmb_type == "chunked":
        return wmb.WholeMemoryMemoryType.MtChunked
    elif str_wmb_type == "distributed":
        return wmb.WholeMemoryMemoryType.MtDistributed
    elif str_wmb_type == "hierarchy":
        return wmb.WholeMemoryMemoryType.MtHierarchy
    else:
        raise ValueError(
            "WholeMemory type %s not supported, should be (continuous, chunked, distributed, hierarchy)"
            % (str_wmb_type,)
        )


def str_to_wmb_wholememory_log_level(str_log_level: str):
    if str_log_level == "error":
        return wmb.WholeMemoryLogLevel.LevError
    elif str_log_level == "warn":
        return wmb.WholeMemoryLogLevel.LevWarn
    elif str_log_level == "info":
        return wmb.WholeMemoryLogLevel.LevInfo
    elif str_log_level == "debug":
        return wmb.WholeMemoryLogLevel.LevDebug
    elif str_log_level == "trace":
        return wmb.WholeMemoryLogLevel.LevTrace
    else:
        raise ValueError(
            "WholeMemory log level %s not supported, shold be (error, warn, info, debug, trace)"
            % (str_log_level,)
        )


def str_to_wmb_wholememory_location(str_wmb_location: str):
    if str_wmb_location == "cuda":
        return wmb.WholeMemoryMemoryLocation.MlDevice
    elif str_wmb_location == "cpu":
        return wmb.WholeMemoryMemoryLocation.MlHost
    else:
        raise ValueError(
            "WholeMemory location %s not supported, should be (cuda, cpu)"
            % (str_wmb_location,)
        )


def str_to_wmb_wholememory_access_type(str_wmb_access: str):
    if str_wmb_access == "readonly" or str_wmb_access == "ro":
        return wmb.WholeMemoryAccessType.AtReadOnly
    elif str_wmb_access == "readwrite" or str_wmb_access == "rw":
        return wmb.WholeMemoryAccessType.AtReadWrite
    else:
        raise ValueError(
            "WholeMemory access %s not supported, should be (readonly, ro, readwrite, rw)"
            % (str_wmb_access,)
        )


def str_to_wmb_wholememory_optimizer_type(str_wmb_optimizer: str):
    if str_wmb_optimizer == "sgd":
        return wmb.WholeMemoryOptimizerType.OptSgd
    elif str_wmb_optimizer == "adam":
        return wmb.WholeMemoryOptimizerType.OptLazyAdam
    elif str_wmb_optimizer == "adagrad":
        return wmb.WholeMemoryOptimizerType.OptAdaGrad
    elif str_wmb_optimizer == "rmsprop":
        return wmb.WholeMemoryOptimizerType.OptRmsProp
    else:
        raise ValueError(
            "WholeMemory optimizer %s not supported, should be (sgd, adam, adagrad, rmsprop)"
            % (str_wmb_optimizer,)
        )


def str_to_wmb_wholememory_distributed_backend_type(str_wmb_distributed_backend: str):
    if str_wmb_distributed_backend == "nccl":
        return wmb.WholeMemoryDistributedBackend.DbNCCL
    elif str_wmb_distributed_backend == "nvshmem":
        return wmb.WholeMemoryDistributedBackend.DbNVSHMEM
    else:
        raise ValueError(
            "WholeMemory str_wmb_distributed_backend %s not supported, should be (nccl, nvshmem)"
            % (str_wmb_distributed_backend,)
        )


def wholememory_distributed_backend_type_to_str(distributed_backend: wmb.WholeMemoryDistributedBackend):
    if distributed_backend == wmb.WholeMemoryDistributedBackend.DbNCCL:
        return "nccl"
    elif distributed_backend == wmb.WholeMemoryDistributedBackend.DbNVSHMEM:
        return "nvshmem"
    else:
        raise ValueError(
            "WholeMemory distributed_backend  not supported, should be (DbNCCL, DbNVSHMEM)"
        )


def get_part_file_name(prefix: str, part_id: int, part_count: int):
    return "%s_part_%d_of_%d" % (prefix, part_id, part_count)


def get_part_file_list(prefix: str, part_count: int):
    filelist = []
    for part_id in range(part_count):
        filelist.append("%s_part_%d_of_%d" % (prefix, part_id, part_count))
    return filelist
