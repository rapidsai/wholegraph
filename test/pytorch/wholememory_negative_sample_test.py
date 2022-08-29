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

from time import time

import os

import torch
from mpi4py import MPI

from wholegraph.torch import wholegraph_pytorch as wg
from wg_torch.wm_tensor import *

comma = MPI.COMM_WORLD
size = comma.Get_size()
rank = comma.Get_rank()

graph_num_nodes = 10000
graph_num_edges = 2000000
negative_sample_count = 1
target_nodes_num = 512

os.environ["RANK"] = str(comma.Get_rank())
os.environ["WORLD_SIZE"] = str(comma.Get_size())
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "12335"
wg.init_lib()
torch.cuda.set_device(rank)
torch.distributed.init_process_group(backend="nccl", init_method="env://")
wm_comm = create_global_communicator(comma.Get_rank(), comma.Get_size())
csr_row_ptr = create_wm_tensor(
    wm_comm,
    [
        graph_num_nodes + 1,
    ],
    [],
    torch.int64,
    WmTensorType.HOST,
)
csr_col_ind = create_wm_tensor(
    wm_comm,
    [
        graph_num_edges,
    ],
    [],
    torch.int64,
    WmTensorType.HOST,
)
if rank == 0:
    tmp_tensor = torch.randint(
        1, graph_num_edges, (graph_num_nodes - 1,), dtype=torch.int64
    )
    zero_tensor = torch.tensor([0])
    full_num_edge_tensor = torch.tensor([graph_num_edges])
    a = torch.cat((tmp_tensor, zero_tensor))
    a = torch.cat((a, full_num_edge_tensor))
    sorted, indices = torch.sort(a)
    csr_row_ptr[:] = sorted
    csr_col_ind[:] = torch.randint(
        0,
        graph_num_nodes,
        (graph_num_edges,),
        dtype=torch.int64,
        device=torch.device("cpu"),
    )
wg.barrier(wm_comm)
comma.barrier()
print("rank=%d, csr_row_ptr=%s" % (rank, csr_row_ptr))
print("rank=%d, csr_col_ind=%s" % (rank, csr_col_ind))
target_node_tensor = torch.randint(
    0,
    graph_num_nodes,
    (target_nodes_num,),
    dtype=torch.int64,
    device=torch.device("cuda"),
)
print("rank=%d, target_node_tensor=%s" % (rank, target_node_tensor))
torch.cuda.synchronize()
wg.barrier(wm_comm)
comma.barrier()

sampled_negative_nodes = torch.ops.wholegraph.per_source_uniform_negative_sample(
    target_node_tensor, csr_row_ptr, csr_col_ind, graph_num_nodes, negative_sample_count
)
print("rank=%d, sampled_negative_nodes=%s" % (rank, sampled_negative_nodes))
torch.cuda.synchronize()
comma.barrier()

# check result
check = True
for i, src_node in enumerate(target_node_tensor):
    for j in range(negative_sample_count):
        negative_dst_node = sampled_negative_nodes[i * negative_sample_count + j]
        neighbor_node_start = csr_row_ptr[src_node]
        neighbor_node_end = csr_row_ptr[src_node + 1]
        for id in range(neighbor_node_start, neighbor_node_end):
            neighbor_node = csr_col_ind[id]
            if neighbor_node == negative_dst_node:
                check = False
                print(
                    "Wrong res, src_id = %ld, negative_id = %ld "
                    % (src_node, negative_dst_node)
                )
                break
        if not check:
            break

if check:
    print("Right result.")

comma.barrier()
start_time = time()
sampled_negative_nodes = torch.ops.wholegraph.per_source_uniform_negative_sample(
    target_node_tensor, csr_row_ptr, csr_col_ind, graph_num_nodes, negative_sample_count
)
torch.cuda.synchronize()
comma.barrier()
end_time = time()
time_second = end_time - start_time
print("rank=%d, time=%f s" % (rank, time_second))
