import torch
from wholememory.torch import wholememory_pytorch as wm

wm.init_lib()
wm.mp_init(0, 1)
a=wm.create_tensor([2,3], [], torch.float32, False, [])
b=wm.create_tensor([4,5], [], torch.float32, True, [])
torch.cuda.set_device(0)
b1=b+1
print('b1=')
print(b1)
bb=wm.get_tensor_view(b, torch.device('cpu'))
bb1=bb+1
print('bb1=')
print(bb1)
del a, b, bb
print('Finalizing...')
wm.finalize_lib()
print('Exiting...')
