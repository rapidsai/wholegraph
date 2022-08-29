# Copyright (c) 2022, NVIDIA CORPORATION.
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

import os

import torch
from mpi4py import MPI
from wg_torch import comm as comm
from wg_torch.wm_tensor import *
from wg_torch import embedding_ops as embedding_ops

from wholegraph.torch import wholegraph_pytorch as wg


def test_gather_one_case(
    wm_comm: int, entry_count: int, embedding_dim: int, gather_count: int
):
    if comm.get_rank() == 0:
        print(
            "  Testing Gathering %d vectors from Embedding (%d, %d)..."
            % (gather_count, entry_count, embedding_dim)
        )
    nccl_embedding = create_wm_tensor(
        wm_comm,
        [entry_count, embedding_dim],
        strides=[],
        tensor_dtype=torch.float32,
        wm_tensor_type=WmTensorType.NCCL,
    )

    local_start, local_count, _, _ = get_partition_plan(wm_comm, entry_count)
    local_embedding = get_local_tensor(nccl_embedding)
    assert local_count == local_embedding.shape[0]
    embedding_dim = local_embedding.shape[1]
    embedding_index_data = (
        torch.arange(
            local_start,
            local_start + local_count,
            device=local_embedding.device,
            dtype=local_embedding.dtype,
        )
        .view(local_count, 1)
        .expand(local_count, embedding_dim)
    )
    embedding_pos_data = (
        torch.arange(
            0,
            embedding_dim * 3,
            3.0,
            device=local_embedding.device,
            dtype=local_embedding.dtype,
        )
        .view(1, embedding_dim)
        .expand(local_count, embedding_dim)
    )

    local_embedding.copy_(embedding_index_data + embedding_pos_data)
    torch.cuda.synchronize()

    sparse_indices = torch.randint(
        0, entry_count, (gather_count,), device="cuda", dtype=torch.int32
    )
    target_index_data = (
        sparse_indices.float().view(gather_count, 1).expand(gather_count, embedding_dim)
    )
    target_pos_data = (
        torch.arange(
            0,
            embedding_dim * 3,
            3.0,
            device=target_index_data.device,
            dtype=target_index_data.dtype,
        )
        .view(1, embedding_dim)
        .expand(gather_count, embedding_dim)
    )
    target_gather_value = target_index_data + target_pos_data

    wg.barrier(wm_comm)
    torch.cuda.synchronize()

    gather_result = embedding_ops.EmbeddingLookupFn.apply(
        sparse_indices, nccl_embedding, None
    )

    if not torch.allclose(target_gather_value, gather_result):
        print("Not all close")
        raise AssertionError()

    wg.barrier(wm_comm)

    del nccl_embedding

    if comm.get_rank() == 0:
        print(
            "  => Gathering %d vectors from Embedding (%d, %d) test passed."
            % (gather_count, entry_count, embedding_dim)
        )


def test_gather(wm_comm):
    if comm.get_rank() == 0:
        print("Gather testing...")
    test_gather_one_case(wm_comm, 1000, 32, 100)
    test_gather_one_case(wm_comm, 1000000, 128, 32432)


def test_scatter_one_case(wm_comm: int, entry_count: int, embedding_dim: int):
    if comm.get_rank() == 0:
        print(
            "  Testing Scattering vectors to Embedding (%d, %d)..."
            % (entry_count, embedding_dim)
        )
    nccl_embedding = create_wm_tensor(
        wm_comm,
        [entry_count, embedding_dim],
        strides=[],
        tensor_dtype=torch.float32,
        wm_tensor_type=WmTensorType.NCCL,
    )

    local_start, local_count, _, _ = get_partition_plan(wm_comm, entry_count)
    local_embedding = get_local_tensor(nccl_embedding)
    local_embedding.zero_()
    assert local_count == local_embedding.shape[0]
    embedding_dim = local_embedding.shape[1]
    rank = wg.get_rank(wm_comm)
    size = wg.get_size(wm_comm)
    arange_indices = torch.arange(
        rank, entry_count, size, dtype=torch.int32, device="cuda"
    )
    scatter_count = arange_indices.shape[0]
    perm_indice = torch.randperm(scatter_count, device="cuda")
    scatter_indices = arange_indices[perm_indice]
    embedding_index_data = (
        scatter_indices.float()
        .view(scatter_count, 1)
        .expand(scatter_count, embedding_dim)
    )
    embedding_pos_data = (
        torch.arange(
            0,
            embedding_dim * 3,
            3.0,
            device=local_embedding.device,
            dtype=local_embedding.dtype,
        )
        .view(1, embedding_dim)
        .expand(scatter_count, embedding_dim)
    )
    scatter_data = embedding_index_data + embedding_pos_data
    embedding_index_data = (
        torch.arange(
            local_start,
            local_start + local_count,
            device=local_embedding.device,
            dtype=local_embedding.dtype,
        )
        .view(local_count, 1)
        .expand(local_count, embedding_dim)
    )
    embedding_pos_data = (
        torch.arange(
            0,
            embedding_dim * 3,
            3.0,
            device=local_embedding.device,
            dtype=local_embedding.dtype,
        )
        .view(1, embedding_dim)
        .expand(local_count, embedding_dim)
    )
    target_local_embedding = embedding_index_data + embedding_pos_data
    torch.cuda.synchronize()
    wg.barrier(wm_comm)
    embedding_ops.scatter_nograd(scatter_data, scatter_indices, nccl_embedding)

    if not torch.allclose(target_local_embedding, local_embedding):
        print("Not all close")
        raise AssertionError()

    torch.cuda.synchronize()
    wg.barrier(wm_comm)

    del nccl_embedding

    if comm.get_rank() == 0:
        print(
            "  => Scattering vectors to Embedding (%d, %d) test passed."
            % (entry_count, embedding_dim)
        )


def test_scatter(wm_comm):
    if comm.get_rank() == 0:
        print("Scatter testing...")
    test_scatter_one_case(wm_comm, 1000, 32)
    test_scatter_one_case(wm_comm, 1000000, 128)


def test_exchange_embedding_grads_one_case(
    wm_comm, entry_count: int, embedding_dim: int, grad_count: int
):
    if comm.get_rank() == 0:
        print(
            "  Testing Update Embedding (%d, %d) with gradient (%d, %d)..."
            % (entry_count, embedding_dim, grad_count, embedding_dim)
        )
    sparse_indices = torch.randint(0, entry_count, (grad_count,), device="cuda")
    sparse_grads = torch.randint(
        -100, 100, (grad_count, embedding_dim), device="cuda"
    ).float()
    (
        local_sparse_indice,
        local_sparse_grad,
    ) = torch.ops.wholegraph.exchange_embedding_grads(
        sparse_indices.int(), sparse_grads, entry_count, wm_comm
    )
    sparse_indices_list = [
        torch.zeros_like(sparse_indices) for _ in range(comm.get_world_size())
    ]
    sparse_grads_list = [
        torch.zeros_like(sparse_grads) for _ in range(comm.get_world_size())
    ]
    torch.distributed.all_gather(sparse_indices_list, sparse_indices)
    torch.distributed.all_gather(sparse_grads_list, sparse_grads)
    all_indices = torch.cat(sparse_indices_list)
    all_grads = torch.cat(sparse_grads_list)

    entry_per_rank = (entry_count + comm.get_world_size() - 1) // comm.get_world_size()
    rank_entry_start = entry_per_rank * comm.get_rank()
    rank_entry_end = min(entry_per_rank * (comm.get_rank() + 1), entry_count)
    start_mask = all_indices >= rank_entry_start
    end_mask = all_indices < rank_entry_end
    mask = start_mask * end_mask
    indices_in_rank = torch.masked_select(all_indices, mask)
    mask_indice = torch.masked_select(
        torch.arange(0, all_indices.shape[0], dtype=torch.long, device="cuda"), mask
    )
    grads_in_rank = all_grads[mask_indice]
    all_scatter_result = torch.zeros(
        (entry_count, embedding_dim), dtype=torch.float32, device="cuda"
    )
    indices_in_rank_expand = indices_in_rank.view(indices_in_rank.shape[0], 1).expand(
        -1, embedding_dim
    )
    all_scatter_result.scatter_add_(0, indices_in_rank_expand.long(), grads_in_rank)
    ref_local_indice = torch.unique(indices_in_rank)
    ref_local_grad = all_scatter_result[ref_local_indice.long()]
    ref_local_indice = ref_local_indice - rank_entry_start
    if not torch.equal(local_sparse_indice, ref_local_indice):
        print(
            "indice not same: Ref: \n%s\nNCCLRes: \n%s\n"
            % (ref_local_indice, local_sparse_indice)
        )
        raise AssertionError()
    if not torch.allclose(local_sparse_grad, ref_local_grad):
        print("Not all close")
        raise AssertionError()
    comm.synchronize()
    if comm.get_rank() == 0:
        print(
            "  => Update Embedding (%d, %d) with gradient (%d, %d) test passed."
            % (entry_count, embedding_dim, grad_count, embedding_dim)
        )


def test_exchange_embedding_grads(wm_comm):
    if comm.get_rank() == 0:
        print("Exchange embedding gradients testing...")
    test_exchange_embedding_grads_one_case(wm_comm, 1000, 32, 100)
    test_exchange_embedding_grads_one_case(wm_comm, 100000, 127, 1234)
    test_exchange_embedding_grads_one_case(wm_comm, 100000, 129, 12345)
    test_exchange_embedding_grads_one_case(wm_comm, 1000000, 128, 32432)


def test_all(wm_comm):
    test_gather(wm_comm)
    test_scatter(wm_comm)
    test_exchange_embedding_grads(wm_comm)


wg.init_lib()
torch.set_num_threads(1)
comma = MPI.COMM_WORLD
shared_comma = comma.Split_type(MPI.COMM_TYPE_SHARED)
os.environ["RANK"] = str(comma.Get_rank())
os.environ["WORLD_SIZE"] = str(comma.Get_size())
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "12335"
local_rank = shared_comma.Get_rank()
local_size = shared_comma.Get_size()

dev_count = torch.cuda.device_count()
assert dev_count > 0
assert local_size <= dev_count
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="nccl", init_method="env://")

wm_comm = create_global_communicator(comma.Get_rank(), comma.Get_size())
torch.distributed.barrier()

test_all(wm_comm)

torch.distributed.barrier()

wg.finalize_lib()
