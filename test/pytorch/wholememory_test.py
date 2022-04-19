import torch
from mpi4py import MPI
from wholememory.torch import wholememory_pytorch as wm
from time import sleep, time

comma = MPI.COMM_WORLD
size = comma.Get_size()
rank = comma.Get_rank()

embedding_dim = 512
vocab_size = 10000000
gather_token_count = 1000000

wm.init_lib()
torch.cuda.set_device(rank)
wm.mp_init(rank, size)
b=wm.create_tensor([vocab_size, embedding_dim], [], torch.float32, True, [])
#total_size, offset = wm.aggregate_size((rank + 3) * 4, [])
#print("rank=%d, total_size=%d, offset=%d" % (rank, total_size, offset))
print("rank=%d, b=%s" % (rank, b))
wm.barrier([])
bb=wm.get_tensor_view(b, torch.device('cpu'))
bb[rank, :] = torch.ones([1, embedding_dim]) * rank
wm.barrier([])
print("rank=%d, bb=%s" % (rank, bb))
bb1=bb+1
print("rank=%d, bb1=%s" % (rank, bb1))
b1=b+1
print("rank=%d, b1=%s" % (rank, b1[:8, :]))
del bb1
del b1
a=wm.create_tensor([vocab_size, embedding_dim], [], torch.float32, False, [])
print("rank=%d, a=%s" % (rank, a))
rank_start = int(vocab_size * rank / size)
rank_end = int(vocab_size * (rank + 1) / size)
a[rank_start:rank_end, :] = torch.rand((rank_end - rank_start, embedding_dim), dtype=torch.float32)
gather_idx = torch.randint(0, vocab_size, (gather_token_count, ), device=torch.device('cuda'))
print('gather_idx=%s' % (gather_idx, ))
torch.cuda.synchronize()
#gather_vec = torch.nn.functional.embedding(gather_idx, a)
gather_vec = torch.ops.wholememory.gather(gather_idx, a, a.dtype)
gather_vec2 = torch.ops.wholememory.gather(gather_idx, a, a.dtype)
gather_vec3 = torch.ops.wholememory.gather(gather_idx, a, a.dtype)
torch.cuda.synchronize()
del gather_vec
del gather_vec2
del gather_vec3
torch.cuda.synchronize()
comma.barrier()
start_time = time()
#gather_vec = torch.nn.functional.embedding(gather_idx, a)
#for i in range(1):
#    gather_vec = torch.ops.wholememory.gather(gather_idx, a)
if rank == 0:
    gather_vec = torch.ops.wholememory.gather(gather_idx, a, a.dtype)
torch.cuda.synchronize()
end_time = time()
comma.barrier()
time_second = end_time - start_time
gather_size = gather_token_count * embedding_dim * 4
bw = gather_size / time_second / 1e9
if rank == 0:
    print('time=%f s, bw=%f GB/s' % (end_time - start_time, bw))
print('rank=%d, Finalizing...' % (rank, ))
wm.finalize_lib()

