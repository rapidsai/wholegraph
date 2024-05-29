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

import pytest
import pylibwholegraph.binding.wholememory_binding as wmb
from pylibwholegraph.utils.multiprocess import multiprocess_run
from pylibwholegraph.torch.initialize import init_torch_env_and_create_wm_comm
import torch
from functools import partial
from pylibwholegraph.test_utils.test_comm import (
    gen_csr_graph,
    copy_host_1D_tensor_to_wholememory,
    host_get_sample_offset_tensor,
    host_sample_all_neighbors,
    int_to_wholememory_datatype,
    int_to_wholememory_location,
    int_to_wholememory_type,
)
import pylibwholegraph.torch.wholegraph_ops as wg_ops
import random


def unweighte_sample_without_replacement_base(random_values, M, N):
    a = torch.empty((M,), dtype=torch.int32)
    Q = torch.arange(N, dtype=torch.int32)
    for i in range(M):
        a[i] = Q[random_values[i]]
        Q[random_values[i]] = Q[N - i - 1]
    return a


def host_unweighted_sample_without_replacement_func(
    host_csr_row_ptr,
    host_csr_col_ptr,
    center_nodes,
    output_sample_offset_tensor,
    col_id_dtype,
    total_sample_count,
    max_sample_count,
    random_seed,
):
    output_dest_tensor = torch.empty((total_sample_count,), dtype=col_id_dtype)
    output_center_localid_tensor = torch.empty((total_sample_count,), dtype=torch.int32)
    output_edge_gid_tensor = torch.empty((total_sample_count,), dtype=torch.int64)
    center_nodes_count = center_nodes.size(0)

    M = max_sample_count

    warp_count = [
        1,
        1,
        1,
        2,
        2,
        2,
        4,
        4,
        4,
        4,
        4,
        4,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
    ]
    total_items_per_thread = [
        1,
        2,
        3,
        2,
        3,
        3,
        2,
        2,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
    ]
    func_idx = int((max_sample_count - 1) / 32)
    block_threads = warp_count[func_idx] * 32
    items_per_thread = total_items_per_thread[func_idx]

    for i in range(center_nodes_count):
        node_id = center_nodes[i]
        start = host_csr_row_ptr[node_id]
        end = host_csr_row_ptr[node_id + 1]
        neighbor_count = end - start
        N = neighbor_count
        output_id = output_sample_offset_tensor[i]
        gidx = i * block_threads
        if neighbor_count <= max_sample_count:
            for j in range(end - start):
                output_dest_tensor[output_id + j] = host_csr_col_ptr[start + j]
                output_center_localid_tensor[output_id + j] = i
                output_edge_gid_tensor[output_id + j] = start + j
        else:
            random_values = torch.empty((N,), dtype=torch.int32)
            for j in range(block_threads):
                local_gidx = gidx + j
                random_nums = wg_ops.generate_random_positive_int_cpu(
                    random_seed, local_gidx, items_per_thread
                )
                for k in range(items_per_thread):
                    id = k * block_threads + j
                    if id < neighbor_count:
                        if id < M:
                            random_values[id] = random_nums[k] % (N - id)
                        else:
                            random_values[id] = N
            random_sample_ids = unweighte_sample_without_replacement_base(
                random_values, M, N
            )
            for sample_id in range(M):
                output_dest_tensor[output_id + sample_id] = host_csr_col_ptr[
                    start + random_sample_ids[sample_id]
                ]
                output_center_localid_tensor[output_id + sample_id] = i
                output_edge_gid_tensor[output_id + sample_id] = (
                    start + random_sample_ids[sample_id]
                )
    return output_dest_tensor, output_center_localid_tensor, output_edge_gid_tensor


def host_unweighted_sample_without_replacement(
    host_csr_row_ptr,
    host_csr_col_ptr,
    center_nodes,
    max_sample_count,
    col_id_dtype,
    random_seed,
):
    center_nodes_count = center_nodes.size(0)
    output_sample_offset_tensor = host_get_sample_offset_tensor(
        host_csr_row_ptr, center_nodes, max_sample_count
    )
    total_sample_count = output_sample_offset_tensor[center_nodes_count]

    if max_sample_count <= 0:
        return host_sample_all_neighbors(
            host_csr_row_ptr,
            host_csr_col_ptr,
            center_nodes,
            output_sample_offset_tensor,
            col_id_dtype,
            total_sample_count,
        )
    if max_sample_count > 1024:
        raise ValueError(
            "invalid host_unweighted_sample_without_replacement test max_sample_count"
        )

    (
        output_dest_tensor,
        output_center_localid_tensor,
        output_edge_gid_tensor,
    ) = host_unweighted_sample_without_replacement_func(
        host_csr_row_ptr,
        host_csr_col_ptr,
        center_nodes,
        output_sample_offset_tensor,
        col_id_dtype,
        total_sample_count,
        max_sample_count,
        random_seed,
    )

    return (
        output_sample_offset_tensor,
        output_dest_tensor,
        output_center_localid_tensor,
        output_edge_gid_tensor,
    )


def routine_func(world_rank: int, world_size: int, **kwargs):
    wm_comm, _ = init_torch_env_and_create_wm_comm(
        world_rank, world_size, world_rank, world_size
    )
    wm_comm = wm_comm.wmb_comm
    host_csr_row_ptr = kwargs["host_csr_row_ptr"]
    host_csr_col_ptr = kwargs["host_csr_col_ptr"]
    graph_node_count = kwargs["graph_node_count"]
    graph_edge_count = kwargs["graph_edge_count"]
    max_sample_count = kwargs["max_sample_count"]
    center_node_count = kwargs["center_node_count"]
    center_node_dtype = kwargs["center_node_dtype"]
    int_col_id_dtype = kwargs["col_id_dtype"]
    int_wholememory_location = kwargs["wholememory_location"]
    int_wholememory_type = kwargs["wholememory_type"]
    need_center_local_output = kwargs["need_center_local_output"]
    need_edge_output = kwargs["need_edge_output"]

    world_rank = wm_comm.get_rank()
    world_size = wm_comm.get_size()

    col_id_dtype = int_to_wholememory_datatype(int_col_id_dtype)
    wholememory_location = int_to_wholememory_location(int_wholememory_location)
    wholememory_type = int_to_wholememory_type(int_wholememory_type)

    if not wm_comm.support_type_location(wholememory_type, wholememory_location):
        wmb.finalize()
        return

    wm_csr_row_ptr = wmb.create_wholememory_array(
        wmb.WholeMemoryDataType.DtInt64,
        graph_node_count + 1,
        wm_comm,
        wholememory_type,
        wholememory_location,
    )
    wm_csr_col_ptr = wmb.create_wholememory_array(
        col_id_dtype, graph_edge_count, wm_comm, wholememory_type, wholememory_location
    )
    copy_host_1D_tensor_to_wholememory(
        wm_csr_row_ptr, host_csr_row_ptr, world_rank, world_size, wm_comm
    )
    copy_host_1D_tensor_to_wholememory(
        wm_csr_col_ptr, host_csr_col_ptr, world_rank, world_size, wm_comm
    )

    wm_comm.barrier()

    center_node_tensor = torch.randint(
        0, graph_node_count, (center_node_count,), dtype=center_node_dtype
    )
    center_node_tensor_cuda = center_node_tensor.cuda()
    random_seed = random.randint(1, 10000)

    # output_sample_offset_tensor_cuda,
    # output_dest_tensor_cuda,
    # output_center_localid_tensor_cuda,
    # output_edge_gid_tensor_cuda =
    # torch.ops.wholegraph.unweighted_sample_without_replacement(wm_csr_row_ptr.get_c_handle(),
    #                            wm_csr_col_ptr.get_c_handle(),
    #                            center_node_tensor_cuda,
    #                            max_sample_count,
    #                            random_seed)
    output_sample_offset_tensor = None
    output_dest_tensor = None
    output_center_localid_tensor = None
    output_edge_gid_tensor = None
    output_tensors = wg_ops.unweighted_sample_without_replacement(
        wm_csr_row_ptr,
        wm_csr_col_ptr,
        center_node_tensor_cuda,
        max_sample_count,
        random_seed,
        need_center_local_output=need_center_local_output,
        need_edge_output=need_edge_output,
    )
    output_cpu_tensors = tuple(tensor.cpu() for tensor in output_tensors)
    torch_col_id_dtype = torch.int32
    if col_id_dtype == wmb.WholeMemoryDataType.DtInt64:
        torch_col_id_dtype = torch.int64
    (
        output_sample_offset_tensor_ref,
        output_dest_tensor_ref,
        output_center_localid_tensor_ref,
        output_edge_gid_tensor_ref,
    ) = host_unweighted_sample_without_replacement(
        host_csr_row_ptr,
        host_csr_col_ptr,
        center_node_tensor,
        max_sample_count,
        torch_col_id_dtype,
        random_seed,
    )

    if need_edge_output and need_center_local_output:
        (
            output_sample_offset_tensor,
            output_dest_tensor,
            output_center_localid_tensor,
            output_edge_gid_tensor,
        ) = output_cpu_tensors
        assert torch.equal(output_sample_offset_tensor, output_sample_offset_tensor_ref)
        assert torch.equal(output_dest_tensor, output_dest_tensor_ref)
        assert torch.equal(
            output_center_localid_tensor, output_center_localid_tensor_ref
        )
        assert torch.equal(output_edge_gid_tensor, output_edge_gid_tensor_ref)
    elif need_center_local_output:
        (
            output_sample_offset_tensor,
            output_dest_tensor,
            output_center_localid_tensor,
        ) = output_cpu_tensors
        assert torch.equal(output_sample_offset_tensor, output_sample_offset_tensor_ref)
        assert torch.equal(output_dest_tensor, output_dest_tensor_ref)
        assert torch.equal(
            output_center_localid_tensor, output_center_localid_tensor_ref
        )
    elif need_edge_output:
        (
            output_sample_offset_tensor,
            output_dest_tensor,
            output_edge_gid_tensor,
        ) = output_cpu_tensors
        assert torch.equal(output_sample_offset_tensor, output_sample_offset_tensor_ref)
        assert torch.equal(output_dest_tensor, output_dest_tensor_ref)
        assert torch.equal(output_edge_gid_tensor, output_edge_gid_tensor_ref)
    else:
        output_sample_offset_tensor, output_dest_tensor = output_cpu_tensors
        assert torch.equal(output_sample_offset_tensor, output_sample_offset_tensor_ref)
        assert torch.equal(output_dest_tensor, output_dest_tensor_ref)

    wmb.destroy_wholememory_tensor(wm_csr_row_ptr)
    wmb.destroy_wholememory_tensor(wm_csr_col_ptr)
    wmb.finalize()


@pytest.mark.parametrize("graph_node_count", [103])
@pytest.mark.parametrize("graph_edge_count", [1043])
@pytest.mark.parametrize("max_sample_count", [11, -1])
@pytest.mark.parametrize("center_node_count", [13])
@pytest.mark.parametrize("center_node_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("col_id_dtype", [0, 1])
@pytest.mark.parametrize("wholememory_location", ([0, 1]))
@pytest.mark.parametrize("wholememory_type", ([0, 1, 2]))
@pytest.mark.parametrize("need_center_local_output", [True, False])
@pytest.mark.parametrize("need_edge_output", [True, False])
def test_wholegraph_unweighted_sample(
    graph_node_count,
    graph_edge_count,
    max_sample_count,
    center_node_count,
    center_node_dtype,
    col_id_dtype,
    wholememory_location,
    wholememory_type,
    need_center_local_output,
    need_edge_output,
):
    gpu_count = wmb.fork_get_gpu_count()
    assert gpu_count > 0
    csr_col_dtype = torch.int32
    if col_id_dtype == wmb.WholeMemoryDataType.DtInt64:
        csr_col_dtype = torch.int64
    host_csr_row_ptr, host_csr_col_ptr, _ = gen_csr_graph(
        graph_node_count, graph_edge_count, csr_col_dtype=csr_col_dtype
    )
    routine_func_partial = partial(
        routine_func,
        host_csr_row_ptr=host_csr_row_ptr,
        host_csr_col_ptr=host_csr_col_ptr,
        graph_node_count=graph_node_count,
        graph_edge_count=graph_edge_count,
        max_sample_count=max_sample_count,
        center_node_count=center_node_count,
        center_node_dtype=center_node_dtype,
        col_id_dtype=col_id_dtype,
        wholememory_location=wholememory_location,
        wholememory_type=wholememory_type,
        need_center_local_output=need_center_local_output,
        need_edge_output=need_edge_output,
    )
    multiprocess_run(gpu_count, routine_func_partial, True)
