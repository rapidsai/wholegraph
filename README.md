# WholeGraph

WholeGraph is developed to help train large-scale Graph Neural Networks(GNN).

The software stack of WholeGraph shown below:

![WholeGraph](docs/imgs/whole_graph_stack.png)

Based on CUDA and NVIDIA's hardware, a multi-GPU storage called [WholeMemory](docs/WholeMemoryIntroduction.md) is implemented.
WholeMemory is used to support both graph structure and feature embedding storage.
Based on the storage, WholeGraph Ops are built to support GNN applications, including sampling ops, embedding ops and some GNN layer ops
On the other hand, sampled subgraph from WholeGraph can also be converted to support PyG or DGL layers.
So, GNN applications can utilize ops from WholeGraph and other GNN frameworks.

## How to Use

### Environment

#### Hardware

It is suggested to use NVLink systems, like DGX-A100 or similar systems.

#### Software

It is recommended to use our [Dockerfile](Dockerfile)

### Compile

To compile WholeMemory, from source directory:

```shell script
mkdir build
cd build
cmake ../
make -j
```

Or you can build release version by replacing `cmake ../` by `cmake -DCMAKE_BUILD_TYPE=Release ..`, which has slightly better performance.

### Examples

Checkout [GNN example](docs/GNNExample.md) for more details. 