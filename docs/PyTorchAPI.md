<link rel="stylesheet" type="text/css" href="auto-number-title.css" />

# PyTorch API for WholeMemory

## Initialization and Finalization

To use PyTorch APIs, it is needed to import WholeMemory module first by:

```python
from wholememory.torch import wholememory_pytorch as wm
```

As C++ API, to use PyTorch API, the library need to be initialized first, by:
```python
wm.init_lib()
```
before any CUDA call, for example `torch.cuda.set_device`

And after the use of WholeMemory, `finalize_lib` should be called.

For PyTorch API, only multi-process multi-GPU mode is support as it is the most common use case.

It is not supported to use single process multi-GPU mode using Python APIs.
The consideration here is that Python has GIL which limits the performance gain of multi-threading, it seems not needed to use this mode, so it is not provided.

After library initialize, you should get rank and size of the WholeMemory group. Then initialize multi-process multi-GPU mode by
```python
wm.mp_init(rank, size)
```

## Tensor Creation

After all this, WholeMemory can be used to create PyTorch Tensors, by `wm.create_tensor`.
Please be noted that this is a collective operation, all process should call it together with the same parameters.
The first argument is the shape of the tensor, in an array.
The second argument is the stride of the tensor, also in an array, or left it empty to automaticly compute that.
The third argument is the datatype of the tensor.
The forth argument is whether to use host memory, True for host memory, False for device memory.
The fifth argument is the ranks participate in this allocation, empty for all ranks. It is recommended to left it empty.

For example, to allocate a 10000000x512 float32 tensor on device, you can do:
```python
a = wm.create_tensor([10000000, 512], [], torch.float32, False, [])
print("rank=%d, a=%s" % (rank, a))
```
Something like this will be output:
```python
rank=0, a=tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
rank=1, a=tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:1')
...
```
Tensor `a` in all process will share the same underlying storage, each process can access `a` like in their own GPU.
Modifications to `a` by any process can be observed by other processes.
Actually, `a` is stored distributed across all GPUs, each have one partition.
This makes it possible to use large Tensors.

## Tensor view

As Tensors created by WholeMemory are across many GPUs, they are also usable in all these GPUs.
By default, Tensor is on the device it was created, e.g. the device torch.cuda.set_device specified.
To get the view of the Tensor from one of the device, `get_tensor_view` can be used to do this.
For example, the following statement will get the view of Tensor `a` from GPU1.
```python
a1 = wm.get_tensor_view(a, torch.device('cuda:1'))
```
Moreover, if a Tensor is created in host memory, the CPU version of the tensor can also be get using this interface, like:
```python
b_cpu = wm.get_tensor_view(b, torch.device('cpu'))
```
