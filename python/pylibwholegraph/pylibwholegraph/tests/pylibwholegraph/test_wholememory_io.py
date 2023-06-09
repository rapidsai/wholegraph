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
import numpy as np
import os
import random
from functools import partial


gpu_count = None


@pytest.fixture(scope="module", autouse=True)
def module_level_setup_teardown():
    skip_env_flag = os.getenv("TEST_WM_LOAD_STORE")
    skip_test = True
    if skip_env_flag is not None:
        skip_env_flag = skip_env_flag.lower()
        if skip_env_flag == "1" or skip_env_flag == "true" or skip_env_flag == "on":
            skip_test = False
    if skip_test:
        pytest.skip("Skipping load store test due to TEST_WM_LOAD_STORE not set...")

    global gpu_count
    assert gpu_count is None
    gpu_count = wmb.fork_get_gpu_count()
    assert gpu_count > 0
    yield
    gpu_count = None


def load_routine_func(
    world_rank: int,
    world_size: int,
    cpu_embedding_tensor_base,
    file_name_prefix,
    file_part_count,
    embedding_entry_count,
    embedding_dim,
    embedding_stride,
    storage_offset,
):
    wm_comm, _ = init_torch_env_and_create_wm_comm(
        world_rank, world_size, world_rank, world_size
    )
    wm_comm = wm_comm.wmb_comm
    data_type = wmb.WholeMemoryDataType.DtInt
    file_list = [None] * file_part_count

    per_rank_entry = wmb.determine_partition_plan(embedding_entry_count, world_size)
    rank_start_entry = min(per_rank_entry * world_rank, embedding_entry_count)
    rank_end_entry = min(per_rank_entry * (world_rank + 1), embedding_entry_count)
    rank_entry_count = rank_end_entry - rank_start_entry

    reference_local_tensor = cpu_embedding_tensor_base[
        rank_start_entry:rank_end_entry, :
    ].cuda()

    for i in range(file_part_count):
        file_list[i] = "%s_part_%d_of_%d" % (file_name_prefix, i, file_part_count)
    for mt in [
        wmb.WholeMemoryMemoryType.MtContinuous,
        wmb.WholeMemoryMemoryType.MtChunked,
        wmb.WholeMemoryMemoryType.MtDistributed,
    ]:
        for ml in [
            wmb.WholeMemoryMemoryLocation.MlHost,
            wmb.WholeMemoryMemoryLocation.MlDevice,
        ]:
            wholememory_root_tensor = wmb.create_wholememory_matrix(
                data_type,
                embedding_entry_count,
                embedding_dim + storage_offset,
                embedding_stride,
                wm_comm,
                mt,
                ml,
            )
            wholememory_tensor = wholememory_root_tensor.get_sub_tensor(
                [-1, storage_offset], [-1, -1]
            )
            wholememory_tensor.from_filelist(file_list)
            local_tensor, local_offset = wholememory_tensor.get_local_tensor(
                torch_import_from_dlpack,
                wmb.WholeMemoryMemoryLocation.MlDevice,
                world_rank,
            )

            assert local_tensor.dim() == 2
            assert local_tensor.shape[0] == rank_entry_count
            assert local_tensor.shape[1] == embedding_dim

            assert torch.equal(local_tensor, reference_local_tensor)
            del wholememory_tensor
            wmb.destroy_wholememory_tensor(wholememory_root_tensor)

    wmb.finalize()


@pytest.mark.parametrize("file_part_count", [3, 5])
@pytest.mark.parametrize(
    "embedding_entry_count", [1024 * 1024 * 4 + 131, 1024 * 1024 * 6 - 127]
)
@pytest.mark.parametrize("embedding_dim", [16, 31, 33])
@pytest.mark.parametrize("embedding_stride", [16, 32, 64])
@pytest.mark.parametrize("storage_offset", [0, 3])
def test_wholememory_load(
    file_part_count,
    embedding_entry_count,
    embedding_dim,
    embedding_stride,
    storage_offset,
):
    if embedding_stride < storage_offset + embedding_dim:
        pytest.skip(
            "Skipping due to embedding_stride, embedding_dim and storage_offset configuration not valid."
        )
    cpu_embedding_tensor_base = torch.randint(
        -1000000000,
        1000000000,
        (embedding_entry_count, embedding_dim),
        dtype=torch.int,
        device="cpu",
    )
    indices = sorted(
        random.sample(range(1, embedding_entry_count), file_part_count - 1)
    )
    indices.append(embedding_entry_count)
    counts = [0] * file_part_count
    for i in range(file_part_count):
        counts[i] = indices[i] if i == 0 else indices[i] - indices[i - 1]
    splited_tensors = torch.split(cpu_embedding_tensor_base, counts, dim=0)
    file_name_prefix = "pytest_load_temp_file"
    for i in range(file_part_count):
        splited_tensors[i].numpy().tofile(
            "%s_part_%d_of_%d" % (file_name_prefix, i, file_part_count)
        )

    cpu_embedding_tensor_base = cpu_embedding_tensor_base.share_memory_()

    load_routine_func_partial = partial(
        load_routine_func,
        cpu_embedding_tensor_base=cpu_embedding_tensor_base,
        file_name_prefix=file_name_prefix,
        file_part_count=file_part_count,
        embedding_entry_count=embedding_entry_count,
        embedding_dim=embedding_dim,
        embedding_stride=embedding_stride,
        storage_offset=storage_offset,
    )

    global gpu_count
    multiprocess_run(gpu_count, load_routine_func_partial)

    for i in range(file_part_count):
        filename = "%s_part_%d_of_%d" % (file_name_prefix, i, file_part_count)
        assert os.path.isfile(filename)
        os.remove(filename)


def store_routine_func(
    world_rank: int,
    world_size: int,
    file_name_prefix,
    embedding_entry_count,
    embedding_dim,
    embedding_stride,
    storage_offset,
):
    (wm_comm,) = init_torch_env_and_create_wm_comm(
        world_rank, world_size, world_rank, world_size
    )
    wm_comm = wm_comm.wmb_comm
    data_type = wmb.WholeMemoryDataType.DtInt

    mt = wmb.WholeMemoryMemoryType.MtContinuous
    ml = wmb.WholeMemoryMemoryLocation.MlHost

    filename = "%s_part_%d_of_%d" % (file_name_prefix, world_rank, world_size)
    wholememory_root_tensor = wmb.create_wholememory_matrix(
        data_type,
        embedding_entry_count,
        embedding_stride,
        embedding_stride,
        wm_comm,
        mt,
        ml,
    )
    local_root_tensor, local_root_offset = wholememory_root_tensor.get_local_tensor(
        torch_import_from_dlpack, wmb.WholeMemoryMemoryLocation.MlHost, world_rank
    )
    root_data_tensor = torch.IntTensor(
        range(
            embedding_stride * local_root_offset,
            embedding_stride * (local_root_offset + local_root_tensor.shape[0]),
        )
    ).reshape((-1, embedding_stride))
    local_root_tensor.copy_(root_data_tensor)

    wholememory_tensor = wholememory_root_tensor.get_sub_tensor(
        [-1, storage_offset], [-1, storage_offset + embedding_dim]
    )

    wholememory_tensor.to_file(filename)

    wmb.finalize()


@pytest.mark.parametrize(
    "embedding_entry_count", [1024 * 1024 * 4 + 131, 1024 * 1024 * 6 - 127]
)
@pytest.mark.parametrize("embedding_dim", [16, 31, 33])
@pytest.mark.parametrize("embedding_stride", [16, 32, 64])
@pytest.mark.parametrize("storage_offset", [0, 3])
def test_wholememory_store(
    embedding_entry_count, embedding_dim, embedding_stride, storage_offset
):
    if embedding_stride < storage_offset + embedding_dim:
        pytest.skip(
            "Skipping due to embedding_stride, embedding_dim and storage_offset configuration not valid."
        )
    file_name_prefix = "pytest_store_temp_file"
    store_routine_func_partial = partial(
        store_routine_func,
        file_name_prefix=file_name_prefix,
        embedding_entry_count=embedding_entry_count,
        embedding_dim=embedding_dim,
        embedding_stride=embedding_stride,
        storage_offset=storage_offset,
    )

    global gpu_count
    multiprocess_run(gpu_count, store_routine_func_partial)
    embedding_entry_offset = 0
    file_part_count = gpu_count
    for i in range(file_part_count):
        filename = "%s_part_%d_of_%d" % (file_name_prefix, i, file_part_count)
        assert os.path.isfile(filename)
        filesize = os.path.getsize(filename)
        assert filesize % (embedding_dim * 4) == 0
        file_entry_count = filesize // (embedding_dim * 4)
        loaded_np_array = np.fromfile(filename, dtype=np.int32)
        loaded_torch_tensor = torch.from_numpy(loaded_np_array).reshape(
            (-1, embedding_dim)
        )

        reference_tensor = torch.IntTensor(
            range(
                embedding_stride * embedding_entry_offset,
                embedding_stride * (embedding_entry_offset + file_entry_count),
            )
        ).reshape((-1, embedding_stride))
        reference_tensor = reference_tensor[
            :, storage_offset : storage_offset + embedding_dim
        ]
        assert torch.equal(loaded_torch_tensor, reference_tensor)

        embedding_entry_offset += file_entry_count
        os.remove(filename)
    assert embedding_entry_offset == embedding_entry_count
