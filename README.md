# WholeGraph

WholeGraph is developed to help train large-scale Graph Neural Networks(GNN).
WholeGraph provides underlying storage structure called WholeMemory.
WholeMemory is a Tensor like storage and provide multi-GPU support.
It is optimized for NVLink systems like DGX A100 servers.
By working together with cuGraph, cuGraph-Ops, cuGraph-DGL, cuGraph-PyG,
and upstream DGL and PyG, it will be easy to build GNN applications.

## Table of content
- Installation
  - [Getting WholeGraph Packages](./docs/wholegraph/source/installation/getting_wholegraph.md)
  - [Building from Source](./docs/wholegraph/source/installation/source_build.md)
- General
  - [WholeGraph Introduction](./docs/wholegraph/source/basics/wholegraph_intro.md)
- Packages
  - libwholegraph (C/CUDA)
  - pylibwholegraph
- API Docs
  - Python
  - C
- Reference
  - [RAPIDS](https://rapids.ai)
  - [cuGraph](https://github.com/rapidsai/cugraph)
