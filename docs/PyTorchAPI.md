# PyTorch API for WholeMemory

## Initialization and Finalization

To use PyTorch APIs, it is needed to import WholeMemory module first by:

```python
from wholegraph.torch import wholegraph_pytorch as wg
```

As C++ API, to use PyTorch API, the library need to be initialized first, by:
```python
wg.init_lib()
```
before any CUDA call, for example `torch.cuda.set_device`

And after the use of WholeMemory, `finalize_lib` should be called.

For PyTorch API, only multi-process multi-GPU mode is support as it is the most common use case.

It is not supported to use single process multi-GPU mode using Python APIs.
The consideration here is that Python has GIL which limits the performance gain of multi-threading, it seems not needed to use this mode, so it is not provided.

After library initialize, you should create the WholeMemory group. First get unique_id at rank 0
```python
unique_id = wg.get_unique_id()
```
And then broadcast the unique_id to all ranks by other communication channel such as MPI or something else.
Then create communicator at all ranks in the same WholeMemory group.
```python
wm_comm=wg.create_communicator(size, unique_id, rank)
```

## Tensor Creation

After all this, Tensor object can be created using `create_wm_tensor` wrapped in `python/wg_torch/wm_tensor.py`.
Please be noted that this is a collective operation, all process should call it together with the same parameters.
- The first argument is the communicator.
- The second argument is the shape of the tensor, in an array.
- The third argument is the stride of the tensor, also in an array, or left it empty to automaticly compute that.
- The forth argument is the datatype of the tensor.
- The fifth argument is the type to allocate, valid values are WmTensorType.HOST, WmTensorType.DEVICE, WmTensorType.CHUNKED and WmTensorType.NCCL.

The return value of the API depends on the fifth argument.

- If it is WmTensorType.HOST, WmTensorType.DEVICE, a norman PyTorch tensor is returned.
- If it is WmTensorType.CHUNKED, a ChunkedTensor object is returned, which seems like a PyTorch Tensor.
As PyTorch custom ops only support a fixed set of inputs, to be able to use `ChunkedTensor`, we use the pointer of the `ChunkedTensor` for all PyTorch ops.
To get the pointer from `ChunkedTensor`, just call its `get_ptr` method.
`ChunkedTensor` also have `shape`, `stride`, `dtype` attributes, which is similar to PyTorch Tensor.
- If it is WmTensorType.NCCL, a NCCLTensor object is returned, which also seems like a PyTorch Tensor.
As PyTorch custom ops only support a fixed set of inputs, to be able to use `NCCLTensor`, we use the pointer of the `NCCLTensor` for all PyTorch ops.
To get the pointer from `NCCLTensor`, just call its `get_ptr` method.
`NCCLTensor` also have `shape`, `stride`, `dtype` attributes, which is similar to PyTorch Tensor.

For example, to allocate a 10000000x512 float32 tensor on device, you can do:
```python
a = create_wm_tensor(wm_comm, [10000000, 512], [], torch.float32, WmTensorType.DEVICE)
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

## Tensor view

As Tensors created by WholeMemory are across many GPUs, they are also usable in all these GPUs.
By default, Tensor is on the device it was created, e.g. the device torch.cuda.set_device specified.
To get the view of the Tensor from one of the device, `get_tensor_view` can be used to do this.
For example, the following statement will get the view of Tensor `a` from GPU1.
```python
a1 = wg.get_tensor_view(a, torch.device('cuda:1'))
```
Moreover, if a Tensor is created in host memory, the CPU version of the tensor can also be get using this interface, like:
```python
b_cpu = wg.get_tensor_view(b, torch.device('cpu'))
```
