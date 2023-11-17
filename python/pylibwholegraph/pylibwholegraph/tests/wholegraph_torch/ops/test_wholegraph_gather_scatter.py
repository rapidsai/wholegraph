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
from pylibwholegraph.torch.dlpack_utils import torch_import_from_dlpack
import torch
import pylibwholegraph.torch.wholememory_ops as wm_ops


# PYTHONPATH=../:$PYTHONPATH python3 -m pytest ../tests/wholegraph_torch/ops/test_wholegraph_gather_scatter.py -s


def gen_int_embedding(indice_tensor, embedding_dim, output_type):
    indice_count = indice_tensor.shape[0]
    indice_part = (
        indice_tensor.type(torch.int).reshape(indice_count, 1).repeat(1, embedding_dim)
    )
    embedding_part = (
        torch.arange(0, embedding_dim, 1, dtype=torch.int)
        .reshape(1, embedding_dim)
        .repeat(indice_count, 1)
    )
    output = indice_part + embedding_part
    return output.type(output_type)


def scatter_gather_test_cast(
    wm_comm,
    dt,
    mt,
    ml,
    embedding_count,
    embedding_dim,
    indice_count,
    use_python_binding=True,
):
    world_rank = wm_comm.get_rank()
    world_size = wm_comm.get_size()
    print(
        "Rank=%d testing scatter gather with embedding_count=%d, embedding_dim=%d, indice_count=%d, dt=%s, mt=%s, ml=%s"
        % (world_rank, embedding_count, embedding_dim, indice_count, dt, mt, ml)
    )
    wm_embedding = wmb.create_wholememory_matrix(
        dt, embedding_count, embedding_dim, -1, wm_comm, mt, ml
    )

    scatter_indice = torch.arange(
        world_rank, embedding_count, world_size, dtype=torch.int64
    )

    embedding_to_scatter = gen_int_embedding(scatter_indice, embedding_dim, torch.float)
    # print('\nscatter_indice=%s\nembedding_to_scatter=%s' % (scatter_indice, embedding_to_scatter))

    scatter_indice_cuda = scatter_indice.cuda()
    embedding_to_scatter_cuda = embedding_to_scatter.cuda()

    if use_python_binding:
        wm_ops.wholememory_scatter_functor(
            embedding_to_scatter_cuda, scatter_indice_cuda, wm_embedding
        )
    else:
        torch.ops.wholegraph.scatter(
            embedding_to_scatter_cuda, scatter_indice_cuda, wm_embedding.get_c_handle()
        )

    wm_comm.barrier()

    del scatter_indice
    del scatter_indice_cuda
    del embedding_to_scatter
    del embedding_to_scatter_cuda

    local_tensor_cuda, local_start = wm_embedding.get_local_tensor(
        torch_import_from_dlpack, wmb.WholeMemoryMemoryLocation.MlDevice, world_rank
    )

    local_ref_start = min(
        wmb.determine_partition_plan(embedding_count, world_size) * world_rank,
        embedding_count,
    )
    local_ref_end = min(
        wmb.determine_partition_plan(embedding_count, world_size) * (world_rank + 1),
        embedding_count,
    )
    local_ref_count = local_ref_end - local_ref_start
    assert local_start == local_ref_start
    assert local_tensor_cuda.dim() == 2
    assert local_tensor_cuda.shape[0] == local_ref_count
    assert local_tensor_cuda.shape[1] == embedding_dim

    local_tensor = local_tensor_cuda.cpu()
    local_indices = torch.arange(local_ref_start, local_ref_end, dtype=torch.int64)
    local_tensor_ref = gen_int_embedding(local_indices, embedding_dim, torch.float)
    # print('\nlocal_tensor %s =%s\nlocal_tensor_ref %s =%s' % (
    #    local_tensor.shape, local_tensor, local_tensor_ref.shape, local_tensor_ref))
    assert torch.allclose(local_tensor, local_tensor_ref)

    gather_indice = torch.randint(0, embedding_count, (indice_count,), dtype=torch.int)
    gather_indice_cuda = gather_indice.cuda()
    if use_python_binding:
        embedding_after_gather_cuda = wm_ops.wholememory_gather_forward_functor(
            wm_embedding, gather_indice_cuda
        )
    else:
        embedding_after_gather_cuda = torch.ops.wholegraph.gather(
            wm_embedding.get_c_handle(), gather_indice_cuda, None, None
        )
    embedding_after_gather = embedding_after_gather_cuda.cpu()
    ref_embedding_gather = gen_int_embedding(gather_indice, embedding_dim, torch.float)
    # print('\ngather_indice=%s\nembedding_after_gather=%s\nref_embedding_gather=%s' % (
    #    gather_indice, embedding_after_gather, ref_embedding_gather))
    assert torch.allclose(embedding_after_gather, ref_embedding_gather)

    del gather_indice
    del gather_indice_cuda
    del embedding_after_gather
    del embedding_after_gather_cuda
    del ref_embedding_gather

    wmb.destroy_wholememory_tensor(wm_embedding)


def routine_func(world_rank: int, world_size: int):
    wm_comm, _ = init_torch_env_and_create_wm_comm(
        world_rank, world_size, world_rank, world_size
    )
    wm_comm = wm_comm.wmb_comm

    embedding_count = 1024 * 256 * world_size + 3
    embedding_dim = 256
    indice_count = 100001
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
                scatter_gather_test_cast(
                    wm_comm, dt, mt, ml, embedding_count, embedding_dim, indice_count, True
                )
                # scatter_gather_test_cast(wm_comm, dt, mt, ml, embedding_count, embedding_dim, indice_count, False)
    wmb.finalize()


def test_wholegraph_gather_scatter():
    gpu_count = wmb.fork_get_gpu_count()
    assert gpu_count > 0
    multiprocess_run(gpu_count, routine_func)
