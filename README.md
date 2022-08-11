# WholeGraph

WholeGraph is a library to help map host or device memory for GPUs to form a WholeGraph using Unified Memory
 Unified Memory means using memory from different GPUs as a [shared memory](https://en.wikipedia.org/wiki/Shared_memory) system or as a [distributed shared memory system](https://en.wikipedia.org/wiki/Distributed_shared_memory), from single process or multiple different processes.
Application developers don't need to care the underlying storage, it looks like a continuous memory range that every GPU can access.
To get best performance, it is recommended to use DGX-A100, DGX-H100 or similar systems with NVSwitch.

## Why WholeGraph?
Typical scenarios are applications needs full access of large GPU memory that can't fit in single GPU.
Including but not limited to large graph storage or large embedding storage related applications.
For example: classic graph computing tasks with large graphs, large graph neural networks, or large knowledge graph tasks.
To make these applications effective, WholeGraph has several features: 
* **Flexible APIs:** WholeGraph has both C++ API and PyTorch APIs.
For C++ API, both single process multi-GPU mode or multi process mode are supported.
For PyTorch API, as normal use case is multiprocess mode, only this mode is supported.
* **Easy to use**: WholeGraph aims to define simple APIs, including C++ APIs and PyTorch APIs.
The APIs have clear semantics.
C++ API is as easy as Init/Finalize and Malloc/Free, and then use the returned pointer.
For PyTorch API these also applies, moreover, the interface can be normal PyTorch Tensor.
All PyTorch OPs working on Tensors can be accelerated with WholeGraph without development.  
* **High performance**: WholeGraph enables GPU peer access, with the help of NVLink, very high bandwidth can be achieved.

## How to Use

For general concepts in WholeGraph, see [Introduction](docs/Introduction.md)

### Environment

#### Hardware

NVLink systems, like DGX-A100 or similar systems.

#### Software

CUDA >= 11.0
CMake >= 3.14
pybind11-dev

### Compile

To compile WholeGraph, from source directory:

```shell script
mkdir build
cd build
cmake ../
make -j
```

### APIs

Please refer to [C++ API](docs/CppAPI.md) or [PyTorch API](docs/PyTorchAPI.md).

If you are using multi-process multi-GPU version, please also refer to [Chunked API](docs/ChunkedAPI.md).

### Performance

Please refer to [Performance](docs/Performance.md) for performance benchmark results.

### Examples

One applications of WholeGraph is GNN.
WholeGraph can be used as the storage for both graph structure and feature embeddings in GNN applications.
See [GNN example](docs/GNNExample.md) for more details. 