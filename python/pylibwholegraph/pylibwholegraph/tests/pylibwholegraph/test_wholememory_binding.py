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
from pylibwholegraph.utils.multiprocess import multiprocess_run
from pylibwholegraph.torch.initialize import init_torch_env_and_create_wm_comm
from pylibwholegraph.torch.dlpack_utils import torch_import_from_dlpack
import torch


# Run with:
# python3 -m pytest ../tests/pylibwholegraph/test_wholememory_binding.py -s


def single_test_case(wm_comm, mt, ml, malloc_size, granularity):
    world_rank = wm_comm.get_rank()
    print("Rank=%d testing mt=%s, ml=%s" % (world_rank, mt, ml))
    h = wmb.malloc(malloc_size, wm_comm, mt, ml, granularity)
    global_tensor = None
    chunked_tensors = None
    view_device = wmb.WholeMemoryMemoryLocation.MlDevice
    view_device_id = world_rank
    tensor_data_type = wmb.WholeMemoryDataType.DtInt64
    elt_size = 8

    local_tensor, local_offset = h.get_local_flatten_tensor(
        torch_import_from_dlpack, tensor_data_type, view_device, view_device_id
    )
    local_data_torch = torch.arange(
        local_offset, local_offset + local_tensor.shape[0], dtype=torch.int64
    )
    local_tensor.copy_(local_data_torch)

    local_view_tensor, _ = h.get_local_flatten_tensor(
        torch_import_from_dlpack, tensor_data_type, view_device, view_device_id
    )
    assert torch.equal(local_view_tensor.cpu(), local_data_torch)
    del local_data_torch, local_view_tensor

    wm_comm.barrier()

    if mt == wmb.WholeMemoryMemoryType.MtDistributed or (
        mt == wmb.WholeMemoryMemoryType.MtChunked
        and ml == wmb.WholeMemoryMemoryLocation.MlDevice
    ):
        with pytest.raises(ValueError):
            global_tensor, _ = h.get_global_flatten_tensor(
                torch_import_from_dlpack, tensor_data_type, view_device, view_device_id
            )
    else:
        global_tensor, _ = h.get_global_flatten_tensor(
            torch_import_from_dlpack, tensor_data_type, view_device, view_device_id
        )
        global_data_torch = torch.arange(0, malloc_size // elt_size, dtype=torch.int64)
        assert torch.equal(global_tensor.cpu(), global_data_torch)
        del global_data_torch

    if mt == wmb.WholeMemoryMemoryType.MtDistributed:
        with pytest.raises(ValueError):
            chunked_tensors, _ = h.get_all_chunked_flatten_tensor(
                torch_import_from_dlpack, tensor_data_type, view_device, view_device_id
            )
    else:
        chunked_tensors, _ = h.get_all_chunked_flatten_tensor(
            torch_import_from_dlpack, tensor_data_type, view_device, view_device_id
        )
        remote_offset = 0
        for i in range(len(chunked_tensors)):
            remote_data_torch = torch.arange(
                remote_offset,
                remote_offset + chunked_tensors[i].shape[0],
                dtype=torch.int64,
            )
            assert torch.equal(chunked_tensors[i].cpu(), remote_data_torch)
            remote_offset += chunked_tensors[i].shape[0]
            del remote_data_torch

    wmb.free(h)


def routine_func(world_rank: int, world_size: int):
    wm_comm, _ = init_torch_env_and_create_wm_comm(
        world_rank, world_size, world_rank, world_size
    )
    wm_comm = wm_comm.wmb_comm

    single_rank_size = 1024 * 1024 * 1024
    malloc_size = single_rank_size * world_size
    granularity = 256

    print("")

    for mt in [
        wmb.WholeMemoryMemoryType.MtContinuous,
        wmb.WholeMemoryMemoryType.MtChunked,
        wmb.WholeMemoryMemoryType.MtDistributed,
    ]:
        for ml in [
            wmb.WholeMemoryMemoryLocation.MlHost,
            wmb.WholeMemoryMemoryLocation.MlDevice,
        ]:
            if wm_comm.support_type_location(mt, ml):
                single_test_case(wm_comm, mt, ml, malloc_size, granularity)
    wmb.finalize()


def test_dlpack():
    gpu_count = wmb.fork_get_gpu_count()
    assert gpu_count > 0
    multiprocess_run(gpu_count, routine_func)
