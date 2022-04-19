# Chunked API for WholeMemory

If you are using WholeMemory for host memory or just in single process mode, just ignore Chunked API, it it not needed.

Chunked API is used as a temporary workaround fix of CUDA Driver API's performance issue.
When using WholeMemory in multi-process multi-GPU mode, memory handles need to be exported by file descriptors to other processes.
Unfortunately, for now, exporting memory handle by FD wile lead to low performance when the memory footprint is large.
It is not a problem if only single process multi-GPU mode is used.
Instead of CUDA Driver API, Chunked API use CUDA Runtime API, as a result, it is not able to use virtual memory management APIs.
So the memory created by Chunked API is many "Chunks", one in each GPU. This can avoid the issue caused by CUDA Driver API.
But also leads to ugly interface...
So after the issue is fixed, this API will be dropped.

## Basic API

To use Chunked APIs, multi-process multi-GPU mode should be initialized, either by C++ API or Python API.

### C++ API

The APIs for WholeChunkedMemory starts with `Wcmmp` which stands for WholeChunkedMemory Multi-Process.
We define `WholeChunkedMemory_t` as the type for WholeChunkedMemory.
As there is no performance problem for host memory or in single process mode, 
there should be only multi-process device memory API.
So the APIs are `WcmmpMalloc` and `WcmmpFree`, different form WholeMemory's C++ API, the memory is represented by `WholeChunkedMemory_t`.
And `min_granularity` can be specified to set the minimum granularity of a single chunk.
More explanations can be found in file `include/whole_chunked_memory.h`.

Different from WholeMemory represented by pointer which is unique for different devices, `WholeChunkedMemory` use `WholeChunkedMemoryHandle` to access the memory, which is different in different devices.
`GetDeviceChunkedHandle` API can be called to get the `WholeChunkedMemoryHandle` for a specific device.

### PyTorch API

For better use WholeChunkedMemory in PyTorch, we wrapped it into a Python class `ChunkedTensor`, which seems like a PyTorch Tensor.
As PyTorch custom ops only support a fixed set of inputs, to be able to use `ChunkedTensor`, we use the pointer of the `ChunkedTensor` for all PyTorch ops.
To get the pointer from `ChunkedTensor`, just call its `get_ptr` method.

`ChunkedTensor` also have `shape`, `stride`, `dtype` attributes, which is similar to PyTorch Tensor.

### Custom op notes

When `ChunkedTensor` is used, adaptions should be made to work on `ChunkedTensor` as the underlying memory is not continuous.

To make the memory looks like continuous memory, we introduced a template class `PtrGen` to generate pointer for a memory offset.
`PtrGen` class is defined in `include/whole_chunked_memory.cuh`

If the underlying memory is continuous, for example `int*` memory, the PtrGen object can be declared like:
```cpp
    PtrGen<int, int> ptr_gen(ptr);
```

If the underlying memory is WholeChunkedMemory, each element also have `int` type, the PtrGen object should be declared like:
```cpp
    PtrGen<const whole_memory::WholeChunkedMemoryHandle, int> ptr_gen(handle);
```

After `ptr_gen` is constructed, `ptr_gen.At(offset)` will return the pointer to storage at `offset`.

So it is recommended to write template kernels which works for both normal continuous memory and WholeChunkedMemory by the use of PtrGen.
