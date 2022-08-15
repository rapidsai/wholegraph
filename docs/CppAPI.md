# C++ API for WholeMemory

## Library initialization / finalization APIs

To use WholeMemory, `WholeMemoryInit` should be called before any CUDA call, as it will initialize CUDA driver API.

After all the work is done, `WholeMemoryFinalize` should be called.

## Single process multi-GPU APIs

Between `WholeMemoryInit` and `WholeMemoryFinalize` API calls, single process multi-GPU APIs can be used.
All the single process multi-GPU APIs start with `Wmsp` which stands for WholeMemory Single Process.

The basic APIs for single process multi-GPU mode WholeMemory APIs should be as simple as `WmspMalloc` and `WmspFree`. And it did.
Only another API is `WmspMallocHost`, whose `free` API is also `WmspFree`.

For `WmspMalloc`, `dev_list` and `dev_count` can be specified to set the GPUs on which to allocate the memory.
Or leave it to default `nullptr` or `0` to use all GPUs.

## Multi-process multi-GPU APIs

To use multi-process multi-GPU APIs, WholeMemory library should first be initialized with `WholeMemoryInit`.
Then multi-process multi-GPU environment should be initialized.

But before that, you should set cuda current device, for example for CUDA runtime API, that is `cudaSetDevice`.
WholeMemory will get current GPU device during multi-process multi-GPU environment initialization.

Another preparation work is to get current GPU's rank and group size of the WholeMemory allocation.
Note that it may be different from other ranks like get from MPI_Comm_rank.
For example when using WholeMemory in a multi-node environment where each node has its own version of memory, local_rank should be used instead of world rank.

With CUDA device set and rank and size prepared, a BootstrapCommunicator should be created to define a group of GPUs who will access each other's memory.
This can be done by first call `WmmpGetUniqueId` at rank 0 of the group to generate a unique ID of the communicator, and then broadcast the unique ID to all ranks in the same group.
Then all ranks should call `WmmpCreateCommunicator` using the unique ID to create the communicator. 
The created communicator can be destroyed by `WmmpDestroyCommunicator`, but be noted that make sure all the WholeMemory are freed before doing this. Or you can just leave the communicator for the library to do the clean up.

After all the work is done, `WmmpFinalize` can be called to stop multi-process multi-GPU mode.
Or just call `WholeMemoryFinalize` if WholeMemory is not needed which will automatic finalize both multi-process multi-GPU mode and single process multi-GPU mode.

### WholeMemory type API

All the multi-process multi-GPU APIs start with `Wmmp` which stants for WholeMemory Multi-Process.

Same as single process multi-GPU mode, the basic APIs for multi-process multi-GPU mode WholeMemory APIs are `WmmpMalloc`, `WmmpMallocHost` and `WmmpFree`.

For `WmmpMalloc` and `WmmpMallocHost`, `bootstrap_communicator` can be specified to tell the library which GPUs to do this allocation.
`bootstrap_communicator` are exposed to keep the capability of setting up different WholeMemory between ranks.
Be noted that `WmmpMalloc`, `WmmpMallocHost` and `WmmpFree` are all collective operations that all ranks inside the same communicator should call together.

For more details of the APIs, refer to the header file `include/whole_memory.h` for details.

### WholeChunkedMemory type API

If you are using WholeMemory for host memory or just in single process mode, just ignore Chunked API, it it not needed.

The multi-process multi-GPU APIs use CUDA Driver APIs to build real continuous memory across GPUs.
Unfortunately, for now, using CUDA Driver API may lead to low performance in multi-process environment when memory footprint is large.
Instead of CUDA Driver API, Chunked API use CUDA Runtime API, as a result, it is not able to use virtual memory management APIs.
So the memory created by Chunked API is many "Chunks", one in each GPU. This can get better performance in the case of large memory footprint.
However, this API is not elegant, and may be dropped later if performance is not a concern.

The APIs for WholeChunkedMemory starts with `Wcmmp` which stands for WholeChunkedMemory Multi-Process.
We define `WholeChunkedMemory_t` as the type for WholeChunkedMemory.
As there is no performance problem for host memory or in single process mode, 
there should be only multi-process device memory API.
So the APIs are `WcmmpMalloc` and `WcmmpFree`, different from WholeMemory's C++ API, the memory is represented by `WholeChunkedMemory_t`.
And `min_granularity` can be specified to set the minimum granularity of a single chunk.
More explanations can be found in file `include/whole_chunked_memory.h`.

Different from WholeMemory represented by pointer which is unique for different devices, `WholeChunkedMemory` use `WholeChunkedMemoryHandle` to access the memory, which is different in different devices.
`GetDeviceChunkedHandle` API can be called to get the `WholeChunkedMemoryHandle` for a specific device.

#### Custom op notes for WholeChunkedMemory

To make the memory looks like continuous memory, we introduced a template class `PtrGen` to generate pointer for a memory offset.
`PtrGen` class is defined in `include/whole_chunked_memory.cuh`

If the underlying memory is continuous, for example `int*` memory, the PtrGen object can be declared like:
```cpp
    PtrGen<int, int> ptr_gen(ptr);
```

If the underlying memory is WholeChunkedMemory, each element also have `int` type, the PtrGen object should be declared like:
```cpp
    PtrGen<const whole_graph::WholeChunkedMemoryHandle, int> ptr_gen(handle);
```

After `ptr_gen` is constructed, `ptr_gen.At(offset)` will return the pointer to storage at `offset`.

So it is recommended to write template kernels which works for both normal continuous memory and WholeChunkedMemory by the use of PtrGen.

### WholeNCCLMemory type API

If you want large embedding tables, for example embedding tables across multi-nodes. WholeNCCLMemory is the only one type of WholeMemory that support it.

The APIs for WholeNCCLMemory starts with `Wnmmp` which stands for WholeNCCLMemory Multi-Process.
We define `WholeNCCLMemory_t` as the type for WholeNCCLMemory.
As there it targets to support multi-node embedding tables, there is only multi-process device memory API.
So the APIs are `WnmmpMalloc` and `WnmmpFree`, different from WholeMemory's C++ API, the memory is represented by `WholeNCCLMemory_t`.
And `min_granularity` can be specified to set the minimum granularity of a single rank.
More explanations can be found in file `include/whole_nccl_memory.h`.

### Notes on Multi-process multi-GPU APIs

Most operations on Multi-process multi-GPU APIs like malloc / free are collective, so please be noted to call them collectively.