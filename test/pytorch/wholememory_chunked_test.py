import torch
from mpi4py import MPI
from wholememory.torch import wholememory_pytorch as wm
from time import sleep, time

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

wm.init_lib()
torch.cuda.set_device(rank)
wm.mp_init(rank, size)
c = wm.create_chunked_tensor([vocab_size, embedding_dim], [], torch.float32, [])
cc = c
cc1 = wm.get_sub_chunked_tensor(cc, [0, 64], [])
print("c.shape=%s, c.stride=%s, c.dtype=%s" % (c.shape, c.stride, c.dtype))
print("cc.shape=%s, cc.stride=%s, cc.dtype=%s" % (cc.shape, cc.stride, cc.dtype))
print("cc1.shape=%s, cc1.stride=%s, cc1.dtype=%s" % (cc1.shape, cc1.stride, cc1.dtype))

gather_idx = torch.randint(0, vocab_size, (gather_token_count, ), device=torch.device('cuda'))
print('gather_idx=%s' % (gather_idx, ))
torch.cuda.synchronize()

gather_vec = torch.ops.wholememory.gather_chunked(gather_idx, c.get_ptr(), c.dtype)
gather_vec2 = torch.ops.wholememory.gather_chunked(gather_idx, c.get_ptr(), c.dtype)
gather_vec3 = torch.ops.wholememory.gather_chunked(gather_idx, c.get_ptr(), c.dtype)
torch.cuda.synchronize()
del gather_vec
del gather_vec2
del gather_vec3
torch.cuda.synchronize()
comma.barrier()
start_time = time()
for i in range(1):
    gather_vec = torch.ops.wholememory.gather_chunked(gather_idx, c.get_ptr(), c.dtype)
torch.cuda.synchronize()
end_time = time()
comma.barrier()
time_second = end_time - start_time
gather_size = gather_token_count * embedding_dim * 4
bw = gather_size / time_second / 1e9
print('time=%f s, bw=%f GB/s' % (end_time - start_time, bw))
print('rank=%d, Finalizing...' % (rank, ))

del cc1
del c
print("finalizing lib")
wm.finalize_lib()
print("finalized lib")
