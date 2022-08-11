# WholeGraph GNN Example

WholeGraph can be used to accelerate applications need large high bandwidth memory.
GNN applications with large graphs need large memory, both for graph structure and feature embedding vectors.
WholeGraph can be used to accelerate large GNN training.

Here we use [ogbn-papers100M](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M) as an example:

WholeGraph can be used to store both graph structure and node feature embedding vectors.

## Dataset Preparation

For GNN training, we use PyTorch as the training framework, so we use multi-process chunked mode of WholeGraph.
As in multi-process mode, we use multi-process to load graph structure and embedding together.
This needs to convert the training data into WholeGraph's data format.
This can be simply done by:
```python
from wm_torch import graph_ops as graph_ops
graph_ops.download_and_convert_papers100m(PATH_TO_CONVERTED_DATA, ogb_root_dir)
```
`ogb_root_dir` is the download path of OGB datasets, defaults to `dataset`, the downloaded data will be stored here.

`PATH_TO_CONVERTED_DATA` is the path to store teh converted data.

## Training

After training data is downloaded and converted. Training can be simply done by our `gnn_example.py` script.

To run single node multi-GPU training:

```shell script
WHOLEGRAPH_PATH=...
export PYTHONPATH=${WHOLEGRAPH_PATH}/python:${WHOLEGRAPH_PATH}/build:${PYTHONPATH}
mpirun -np 8 python ${WHOLEGRAPH_PATH}/examples/gnn/gnn_example.py -g PATH_TO_CONVERTED_DATA
```

This script also support multi-node training, but need to set `MASTER_ADDR` and `MASTER_PORT` correctly.

## Performance

We benchmarked the performance using a 3 layer GraphSAGE model with sample count 30,30,30, and batchsize is 1024.
All hidden size is 256, For GAT model, the head count is 4.
Dataset is ogbn-papers100M.
The performance metrics are time per epoch in seconds.  

All the benchmark is done on DGX-A100. The result is shown in the following table.

First two columns are using DGL or PyG's default storage and sampling op. That is CPU storage and sampling.
As we can see, it will take hundreds of seconds for one epoch. We also find that GPU utilization is low in both cases.
With WholeGraph's graph structure storage, sampling, feature storage and gathering, the epoch time time can be reduced to 6.9s using DGL backend and 7.2s using PyG backend.
The result is shown in column WM+DGL and WM+PyG.
We also implemented the forward and backward pass for GCN, GraphSage and GAT model, we also call this implementation WholeGraph Graph (WGG).
The Result is shown in the last column. This can further reduce epoch time.

`gnn_example.py` is the script to reproduce the last 3 columns.

|   Model   |    DGL   |    PyG   |  WG+DGL  |  WG+PyG  |   WG+WGG  |
|   :---:   | :------: | :------: | :------: | :------: | :------: |
|    GCN    |    220   |    359   |    6.7   |    6.7   |    5.3   |
| GraphSage |    274   |    315   |    6.9   |    7.2   |    5.7   |
|    GAT    |    269   |    405   |   24.0   |   45.2   |   21.1   |
