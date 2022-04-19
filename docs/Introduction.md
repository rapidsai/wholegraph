# WholeMemory Introduction

When memory footprint is larger than one GPU's device memory, to make the work run, several approach may be adopted.
One solution is to use host memory, however, in this case the performance may be bounded by PCIe bandwidth.
To get better performance, it is also possible to use GPUs as a [distributed memory](https://en.wikipedia.org/wiki/Distributed_memory) system.
However, this will need explicit communication. Addition efforts need to be made to achieve this.
Which makes it complicated to program and also some runtime overhead like gather/scatter for communication preparation.

WholeMemory provides another way.
The idea of WholeMemory is to use GPUs with NVLink connected as a [shared memory](https://en.wikipedia.org/wiki/Shared_memory) system.
Compared to using GPUs as a distributed memory system, WholeMemory makes it easier to work on large memory.
And also provides very high performance, e.g. 70% ~ 80% of NVLink theoretical bandwidth. 

## WholeMemory Memory memory types

There are two types of memory supported by WholeMemory, one is GPU peer memory, the other is pinned host memory.
When using GPU memory in WholeMemory, both high performance and easy programming can be obtained.
For some cases where large memory capacity is the most important or bandwidth is not a concern, pinned host memory can be used for both CPU and GPU ops to work on this memory.

## WholeMemory working modes

There are two working modes for WholeMemory, one is single process multi-GPU mode, the other is multi-process multi-GPU mode.

### Single process multi-GPU mode

In this mode, all GPUs are used in a single process, typically one CPU thread for one GPU.
In this case WholeMemory will do the underlying memory management and provides simple malloc/free interface.

### Multi-process multi-GPU mode

In this mode, each process is responsible for one GPU, all processes work together to get the work done.
In this case WholeMemory will also do the communication needed when doing memory allocation and free.

## WholeMemory API Types

Both C++ and PyTorch API are supported now. Please read the docs for C++ API or PyTorch API.

## Current workaround for best performance

When using WholeMemory in multi-process multi-GPU mode, it is a pity that sharing GPU memory between process will cause performance degradation now.
This is related to CUDA's virtual memory management API, which may be fixed later.
For now, to get best performance, we also work out a set of workaround APIs, that is Whole Chunked Memory API.
Whole Chunked Memory API breaks the assumption of continuous memory seen by each GPU.
So Ops or kernels working on Whole Chunked Memory should handle this.
To do this more effective, a device API is provided to wrap this temporary fix and continuous memory as a unified PtrGen interface.
When the performance issue of CUDA's virtual memory management API is fixed, this set of API will be deprecated.