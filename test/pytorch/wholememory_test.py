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

embedding_dim = 512
vocab_size = 10000000
gather_token_count = 1000000

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
b = create_wm_tensor(
    wm_comm, [vocab_size, embedding_dim], [], torch.float32, WmTensorType.HOST
)
# total_size, offset = wg.aggregate_size((rank + 3) * 4, [])
# print("rank=%d, total_size=%d, offset=%d" % (rank, total_size, offset))
print("rank=%d, b=%s" % (rank, b))
wg.barrier(wm_comm)
bb = wg.get_tensor_view(b, torch.device("cpu"))
bb[rank, :] = torch.ones([1, embedding_dim]) * rank
wg.barrier(wm_comm)
print("rank=%d, bb=%s" % (rank, bb))
bb1 = bb + 1
print("rank=%d, bb1=%s" % (rank, bb1))
b1 = b + 1
print("rank=%d, b1=%s" % (rank, b1[:8, :]))
del bb1
del b1
a = create_wm_tensor(
    wm_comm, [vocab_size, embedding_dim], [], torch.float32, WmTensorType.DEVICE
)
print("rank=%d, a=%s" % (rank, a))
rank_start = int(vocab_size * rank / size)
rank_end = int(vocab_size * (rank + 1) / size)
a[rank_start:rank_end, :] = torch.rand(
    (rank_end - rank_start, embedding_dim), dtype=torch.float32
)
gather_idx = torch.randint(
    0, vocab_size, (gather_token_count,), device=torch.device("cuda")
)
print("gather_idx=%s" % (gather_idx,))
torch.cuda.synchronize()
# gather_vec = torch.nn.functional.embedding(gather_idx, a)
gather_vec = torch.ops.wholegraph.gather(gather_idx, a, a.dtype)
gather_vec2 = torch.ops.wholegraph.gather(gather_idx, a, a.dtype)
gather_vec3 = torch.ops.wholegraph.gather(gather_idx, a, a.dtype)
torch.cuda.synchronize()
del gather_vec
del gather_vec2
del gather_vec3
torch.cuda.synchronize()
comma.barrier()
start_time = time()
# gather_vec = torch.nn.functional.embedding(gather_idx, a)
# for i in range(1):
#    gather_vec = torch.ops.wholegraph.gather(gather_idx, a)
if rank == 0:
    gather_vec = torch.ops.wholegraph.gather(gather_idx, a, a.dtype)
torch.cuda.synchronize()
end_time = time()
comma.barrier()
time_second = end_time - start_time
gather_size = gather_token_count * embedding_dim * 4
bw = gather_size / time_second / 1e9
if rank == 0:
    print("time=%f s, bw=%f GB/s" % (end_time - start_time, bw))
print("rank=%d, Finalizing..." % (rank,))
wg.finalize_lib()
