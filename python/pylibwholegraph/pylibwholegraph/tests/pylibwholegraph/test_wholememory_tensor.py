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

import pylibwholegraph.binding.wholememory_binding as wmb
from pylibwholegraph.utils.multiprocess import multiprocess_run
from pylibwholegraph.torch.initialize import init_torch_env_and_create_wm_comm


# Run with:
# python3 -m pytest ../tests/pylibwholegraph/test_wholememory_tensor.py -s


def array_test_case(wm_comm, dt, mt, ml, size):
    world_rank = wm_comm.get_rank()
    print(
        "Rank=%d testing array size=%d dt=%s, mt=%s, ml=%s"
        % (world_rank, size, dt, mt, ml)
    )
    wm_array = wmb.create_wholememory_array(dt, size, wm_comm, mt, ml)
    assert wm_array.dtype == dt
    assert wm_array.dim() == 1
    assert len(wm_array.shape) == 1
    assert wm_array.shape[0] == size
    assert len(wm_array.stride()) == 1
    assert wm_array.stride()[0] == 1
    assert wm_array.storage_offset() == 0

    wm_sub_array = wm_array.get_sub_tensor([size // 4], [-1])
    assert wm_sub_array.dtype == dt
    assert wm_sub_array.dim() == 1
    assert wm_sub_array.shape[0] == size - size // 4
    assert wm_sub_array.stride()[0] == 1
    assert wm_sub_array.storage_offset() == size // 4

    wmb.destroy_wholememory_tensor(wm_sub_array)

    wmb.destroy_wholememory_tensor(wm_array)


def matrix_test_case(wm_comm, dt, mt, ml, mat_size):
    world_rank = wm_comm.get_rank()
    print(
        "Rank=%d testing matrix size=%s dt=%s, mt=%s, ml=%s"
        % (world_rank, mat_size, dt, mt, ml)
    )
    wm_matrix = wmb.create_wholememory_matrix(
        dt, mat_size[0], mat_size[1], -1, wm_comm, mt, ml
    )

    assert wm_matrix.dtype == dt
    assert wm_matrix.dim() == 2
    assert len(wm_matrix.shape) == 2
    assert wm_matrix.shape[0] == mat_size[0]
    assert wm_matrix.shape[1] == mat_size[1]
    assert len(wm_matrix.stride()) == 2
    assert wm_matrix.stride()[0] == mat_size[1]
    assert wm_matrix.stride()[1] == 1

    wm_sub_matrix = wm_matrix.get_sub_tensor(
        [mat_size[0] // 3, mat_size[1] // 5], [-1, mat_size[1] // 5 * 3]
    )
    assert wm_sub_matrix.dtype == dt
    assert wm_sub_matrix.dim() == 2
    assert wm_sub_matrix.shape[0] == mat_size[0] - mat_size[0] // 3
    assert wm_sub_matrix.shape[1] == mat_size[1] // 5 * 3 - mat_size[1] // 5
    assert wm_sub_matrix.stride()[0] == mat_size[1]
    assert wm_sub_matrix.stride()[1] == 1
    assert (
        wm_sub_matrix.storage_offset()
        == mat_size[1] // 5 + mat_size[0] // 3 * mat_size[1]
    )

    wmb.destroy_wholememory_tensor(wm_sub_matrix)
    wmb.destroy_wholememory_tensor(wm_matrix)


def routine_func(world_rank: int, world_size: int):
    wm_comm, _ = init_torch_env_and_create_wm_comm(
        world_rank, world_size, world_rank, world_size
    )
    wm_comm = wm_comm.wmb_comm

    single_array_size = 128 * 1024 * 1024 * world_size
    single_matrix_size = (1024 * 1024 * world_size, 128)
    dt = wmb.WholeMemoryDataType.DtFloat

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
                array_test_case(wm_comm, dt, mt, ml, single_array_size)
                matrix_test_case(wm_comm, dt, mt, ml, single_matrix_size)
    wmb.finalize()


def test_wholememory_tensor():
    gpu_count = wmb.fork_get_gpu_count()
    assert gpu_count > 0
    multiprocess_run(gpu_count, routine_func)
