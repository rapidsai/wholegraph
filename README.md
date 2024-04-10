# WholeGraph

WholeGraph is developed to help train large-scale Graph Neural Networks(GNN).
WholeGraph provides underlying storage structure called WholeMemory.
WholeMemory is a Tensor like storage and provide multi-GPU support.
It is optimized for NVLink systems like DGX A100 servers.
By working together with cuGraph, cuGraph-Ops, cuGraph-DGL, cuGraph-PyG,
and upstream DGL and PyG, it will be easy to build GNN applications.

## Table of Contents
- Installation
  - [Getting WholeGraph Packages](https://docs.rapids.ai/api/cugraph/nightly/wholegraph/installation/getting_wholegraph/)
  - [Building from Source](https://docs.rapids.ai/api/cugraph/nightly/wholegraph/installation/source_build/)
- General
  - [WholeGraph Introduction](https://docs.rapids.ai/api/cugraph/nightly/wholegraph/basics/wholegraph_intro/)
- Packages
  - libwholegraph (C/CUDA)
  - pylibwholegraph
- API Docs
  - [Python](https://docs.rapids.ai/api/cugraph/nightly/api_docs/wholegraph/pylibwholegraph/)
  - [C](https://docs.rapids.ai/api/cugraph/nightly/api_docs/wholegraph/libwholegraph/#wholegraph-c-api-documentation)
- Reference
  - [RAPIDS](https://rapids.ai)
  - [cuGraph](https://github.com/rapidsai/cugraph)
