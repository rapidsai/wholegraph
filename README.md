# WholeGraph

WholeGraph is developed to help train large-scale Graph Neural Networks(GNN).
WholeGraph provides underlying storage structure called WholeMemory.
WholeMemory is a Tensor like storage and provide multi-GPU support.
It is optimized for NVLink systems like DGX A100 servers.
By working together with cuGraph, cuGraph-Ops, DGL and PyG, it will be easy to build GNN applications.
