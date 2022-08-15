# WholeMemory Introduction

WholeMemory is the underlying storage library of WholeGraph to help utilize host or device memory of multiple GPUs as a "Whole Memory".
Besides using GPUs as a [distributed memory](https://en.wikipedia.org/wiki/Distributed_memory) system,
WholeMemory supports using memory from different GPUs as a [shared memory](https://en.wikipedia.org/wiki/Shared_memory) system or as a [distributed shared memory system](https://en.wikipedia.org/wiki/Distributed_shared_memory),
from single process or multiple different processes.
When using shared memory approach, application developers don't need to care the underlying storage, it looks like a continuous memory range that every GPU can access.
To get best performance, it is recommended to use DGX-A100 or similar systems with NVSwitch.

Typical applicable scenarios are applications needs access of large GPU memory that can't fit in single GPU.
Including but not limited to large graph storage or large embedding storage related applications.
For example: classic graph computing tasks with large graphs, large graph neural networks, or large knowledge graph tasks.

## WholeMemory Memory memory types

When memory footprint is larger than one GPU's device memory, to make the work run, several approach may be adopted.

One solution is to use host memory, this approach can provide large memory capacity,
however, in this case the performance may be bounded by PCIe bandwidth.

To get better performance, it is also possible to use GPUs as a [distributed memory](https://en.wikipedia.org/wiki/Distributed_memory) system.
For example, communicate using [NCCL](https://developer.nvidia.com/nccl).
This approach can achieve higher performance than host memory, and is also capable of supporting cross-node partitioning.
However, this will need explicit communication. Addition efforts need to be made to achieve this.
Which makes it complicated to program and also some runtime overhead like gather/scatter for communication preparation.

There is also another way.
That is to use GPUs with NVLink connected as a [shared memory](https://en.wikipedia.org/wiki/Shared_memory) system.
Compared to using GPUs as a distributed memory system, this makes it easier to work on large memory.
And also provides very high performance, e.g. 70%+ of NVLink theoretical bandwidth. 

WholeMemory provides all of the above options by different types of memory.

There are logically three types of memory supported by WholeMemory, they are GPU peer memory, pinned host memory, and GPU memory communicating using NCCL.
For GPU peer memory, to get better performance, there are two implementations, so total 4 types of memory in WholeMemory.
The four types of memory types are:

|         Type         | Location | Scalability  |       Performance      |     Capacity     | Continuous Address? | Quasi-Continuous Address? |
|:--------------------:|:--------:|:------------:|:----------------------:|:----------------:|:-------------------:|:-------------------------:|
| WholeMemory (Device) |  device  | single node  |  Best(when not  large) | suggested < 4 GB |        Yes          |             -             |
| WholeMemory (Host)   |   host   | single node  |    limited by PCIe     |    up to 2 TB    |        Yes          |             -             |
| WholeChunkedMemory   |  device  | single node  |         Best           |   up to 640 GB   |         No          |            Yes            |
|   WholeNCCLMemory    |  device  | across nodes |   limited by network   |     No limit     |         No          |             No            |

## WholeMemory working modes

There are two working modes for WholeMemory, one is single process multi-GPU mode, the other is multi-process multi-GPU mode.

### Single process multi-GPU mode

In this mode, all GPUs are used in a single process, typically one CPU thread for one GPU.
In this case WholeMemory will do the underlying memory management and provides simple malloc/free interface.
Only WholeMemory type is supported in this case. That is, you can allocate WholeMemory on Device or Host.

### Multi-process multi-GPU mode

In this mode, each process is responsible for one GPU, all processes work together to get the work done.
In this case the library will also do the communication needed when doing memory allocation and free, using underlying communicator.
So communicator should be created before any malloc / free.
All 4 types of WholeMemory are supported.

## WholeMemory API Types

Both [C++ API](CppAPI.md) and [PyTorch API](PyTorchAPI.md) are supported now. Please read the docs for C++ API or PyTorch API.

## WholeMemory Performance

Please refer to [Performance](Performance.md) for performance benchmark results.

## Notes on WholeChunkedMemory

When using WholeMemory in multi-process multi-GPU mode, it is a pity that sharing large amount of GPU memory between process will cause performance degradation now.
The solution for this case is to use WholeChunkedMemory. 
However, as WholeChunkedMemory has only quasi-continuous address space, Ops or kernels working on WholeChunkedMemory should handle this.

## Notes on WholeNCCLMemory
Using shared memory approach will limit the application to GPUs with P2P access. To support larger memory, we introduce WholeNCCLMemory.
Usually embedding table is the largest component, and the operations on embedding tables are usually gather forward / backward.
So we support embedding storage in WholeNCCLMemory.
