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

embedding_dim = 128
vocab_size = 100000
gather_token_count = 10000
if size == 8:
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
c = create_wm_tensor(
    wm_comm,
    [vocab_size, embedding_dim],
    strides=[],
    tensor_dtype=torch.float32,
    wm_tensor_type=WmTensorType.CHUNKED,
)
cc = c
cc1 = wg.get_sub_chunked_tensor(cc, [0, 64], [])
print("c.shape=%s, c.stride=%s, c.dtype=%s" % (c.shape, c.stride, c.dtype))
print("cc.shape=%s, cc.stride=%s, cc.dtype=%s" % (cc.shape, cc.stride, cc.dtype))
print("cc1.shape=%s, cc1.stride=%s, cc1.dtype=%s" % (cc1.shape, cc1.stride, cc1.dtype))

gather_idx = torch.randint(
    0, vocab_size, (gather_token_count,), device=torch.device("cuda")
)
print("gather_idx=%s" % (gather_idx,))
torch.cuda.synchronize()

gather_vec = torch.ops.wholegraph.gather_chunked(gather_idx, c.get_ptr(), c.dtype)
gather_vec2 = torch.ops.wholegraph.gather_chunked(gather_idx, c.get_ptr(), c.dtype)
gather_vec3 = torch.ops.wholegraph.gather_chunked(gather_idx, c.get_ptr(), c.dtype)
torch.cuda.synchronize()
del gather_vec
del gather_vec2
del gather_vec3
torch.cuda.synchronize()
comma.barrier()
start_time = time()
for i in range(1):
    gather_vec = torch.ops.wholegraph.gather_chunked(gather_idx, c.get_ptr(), c.dtype)
torch.cuda.synchronize()
end_time = time()
comma.barrier()
time_second = end_time - start_time
gather_size = gather_token_count * embedding_dim * 4
bw = gather_size / time_second / 1e9
print("time=%f s, bw=%f GB/s" % (end_time - start_time, bw))
print("rank=%d, Finalizing..." % (rank,))

del cc1
del c
print("finalizing lib")
wg.finalize_lib()
print("finalized lib")
