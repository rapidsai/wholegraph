# C++ API for WholeMemory

## Library initialization / finalization APIs

To use WholeMemory, `WholeMemoryInit` should be called before any cuda call, as it will initialize CUDA driver API.

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

With CUDA device set and rank and size prepared, `WmmpInit` can be called to initialize multi-process multi-GPU environment.
Besides rank and size, there is another `collective_communicator` can be passed, it is used to implement customer communication and usually keep it nullptr is OK.

After all the work is done, `WmmpFinalize` can be called to stop multi-process multi-GPU mode.
Or just call `WholeMemoryFinalize` if WholeMemory is not needed which will automatic finalize both multi-process multi-GPU mode and single process multi-GPU mode.

Between `WmmpInit` and `WmmpFinalize` / `WholeMemoryFinalize`, multi-process multi-GPU APIs can be used.
All the multi-process multi-GPU APIs start with `Wmmp` which stants for WholeMemory Multi-Process.

Same as single process multi-GPU mode, the basic APIs for multi-process multi-GPU mode WholeMemory APIs are `WmmpMalloc`, `WmmpMallocHost` and `WmmpFree`.

For `WmmpMalloc` and `WmmpMallocHost`, `ranks` and `rank_count` can be specified to tell the library which ranks to do this allocation.
`ranks` and `rank_count` are exposed to keep the capability of setting up different WholeMemory between ranks.
Be noted that `WmmpMalloc`, `WmmpMallocHost` and `WmmpFree` are all collective operations that all ranks should call together.
And it is strongly suggested to keep `ranks` and `rank_count` as default `nullptr` and `0`.

## Header file

For more details of the APIs, refer to the header file `include/whole_memory.h` for details.