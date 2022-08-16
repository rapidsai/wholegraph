# WholeGraph GNN Example

WholeMemory can be used to accelerate applications need large high bandwidth memory.
GNN applications with large graphs need large memory, both for graph structure and feature embedding vectors.
WholeGraph is one solution to accelerate large GNN training based on WholeMemory.

## Node Classification Task

Here we use [ogbn-papers100M](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M) as an example:

WholeGraph can be used to store both graph structure and node feature embedding vectors.

### Dataset Preparation

For GNN training, we use PyTorch as the training framework, so we use multi-process chunked mode of WholeMemory.
As in multi-process mode, we use multi-process to load graph structure and embedding together.
This needs to convert the training data into WholeGraph's data format. E.g. generating the graph.
For homograph, the `examples/gnn/gnn_homograph_data_preprocess.py` can be used to convert data and build graph.
```python
WHOLEGRAPH_PATH=...
export PYTHONPATH=${WHOLEGRAPH_PATH}/python:${WHOLEGRAPH_PATH}/build:${PYTHONPATH}
python ${WHOLEGRAPH_PATH}/examples/gnn/gnn_homograph_data_preprocess.py -r ${DATASET_ROOT} -g ogbn-papers100M -p convert
python ${WHOLEGRAPH_PATH}/examples/gnn/gnn_homograph_data_preprocess.py -r ${DATASET_ROOT} -g ogbn-papers100M -p build
```
`DATASET_ROOT` is the download path of OGB datasets, defaults to `dataset`, the downloaded data will be stored here.
The converted data and built graph will be stored under `${DATASET_ROOT}/${NORMALIZED_GRAPH_NAME}/converted`

### Training

After training data is downloaded and converted. Training can be simply done by our `gnn_example_node_classification.py` script.

To run single node multi-GPU training:

```shell script
WHOLEGRAPH_PATH=...
export PYTHONPATH=${WHOLEGRAPH_PATH}/python:${WHOLEGRAPH_PATH}/build:${PYTHONPATH}
mpirun -np 8 python ${WHOLEGRAPH_PATH}/examples/gnn/gnn_example_node_classification.py -r ${DATASET_ROOT} -g ogbn-papers100M
```

This script also support multi-node training, but need to set `MASTER_ADDR` and `MASTER_PORT` correctly.

### Performance

We benchmarked the performance using a 3 layer GraphSAGE model with sample count 30,30,30, and batchsize is 1024.
All hidden size is 256, For GAT model, the head count is 4.
Dataset is ogbn-papers100M.
The performance metrics are time per epoch in seconds.  

All the benchmark is done on DGX-A100. The result is shown in the following table.

First two columns are using DGL or PyG's default storage and sampling op. That is CPU storage and sampling.
As we can see, it will take hundreds of seconds for one epoch. We also find that GPU utilization is low in both cases.
With WholeGraph's graph structure storage, sampling, feature storage and gathering, the epoch time time can be reduced to 6.9s using DGL backend and 7.2s using PyG backend.
The result is shown in column WG+DGL and WG+PyG.
We also implemented the forward and backward pass for GCN, GraphSage and GAT model, we call this implementation WholeGraph(WG).
The Result is shown in the last column. This can further reduce epoch time.

`gnn_example.py` is the script to reproduce the last 3 columns.

|   Model   |    DGL   |    PyG   |  WG+DGL  |  WG+PyG  |    WG    |
|   :---:   | :------: | :------: | :------: | :------: | :------: |
|    GCN    |    220   |    359   |    6.7   |    6.7   |    5.7   |
| GraphSage |    274   |    315   |    6.9   |    7.2   |    6.0   |
|    GAT    |    269   |    405   |   24.0   |   45.2   |   21.1   |

## Link Prediction Task

Here we use [ogbn-citation2](https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2) as an example:

### Dataset Preparation

Same as node classification task, we need to convert the training data into WholeGraph's data format.
This can be simply done by:
```python
WHOLEGRAPH_PATH=...
export PYTHONPATH=${WHOLEGRAPH_PATH}/python:${WHOLEGRAPH_PATH}/build:${PYTHONPATH}
python ${WHOLEGRAPH_PATH}/examples/gnn/gnn_homograph_data_preprocess.py -r ${DATASET_ROOT} -g ogbl-citation2 -p convert
python ${WHOLEGRAPH_PATH}/examples/gnn/gnn_homograph_data_preprocess.py -r ${DATASET_ROOT} -g ogbl-citation2 -p build
```
`DATASET_ROOT` is the download path of OGB datasets, defaults to `dataset`, the downloaded data will be stored here.
The converted data and built graph will be stored under `${DATASET_ROOT}/${NORMALIZED_GRAPH_NAME}/converted`

### Training

After training data is downloaded and converted. Training can be simply done by our `gnn_example.py` script.

To run single node multi-GPU training:

```shell script
WHOLEGRAPH_PATH=...
export PYTHONPATH=${WHOLEGRAPH_PATH}/python:${WHOLEGRAPH_PATH}/build:${PYTHONPATH}
mpirun -np 8 python ${WHOLEGRAPH_PATH}/examples/gnn/gnn_example_link_prediction.py -r ${DATASET_ROOT} -g ogbl-citation2
```

This script also support multi-node training, but need to set `MASTER_ADDR` and `MASTER_PORT` correctly.

### Performance
Here we use the [link prediction example of GraphSAGE](https://github.com/dmlc/dgl/blob/0.8.2/examples/pytorch/graphsage/link_pred.py) from DGL as baseline.
We changed batchsize to 1024.
As DGL use its own MRR calculator, we also add [OGB's evaluator for citation2](https://ogb.stanford.edu/docs/linkprop/#eval).
The DGL version we use is 0.8.2. We evaluated with and without --pure-gpu for DGL's example, which use GPU graph storage and UVA sampling of CPU graph storage.
All model configure and other hyper parameters are tried to be aligned with DGL's example.
The final script is [gnn_example_link_pred.py](../examples/gnn/gnn_example_link_prediction.py)

We train 1 epoch of all 60M edges(double of original 30M citation edges as it is converted to undirected graph).
All test run on DGX-A100 640G. For dgl as the script support only 1 GPU. So only 1 GPU is tested.
For WholeGraph, we run 1 GPU and 8 GPU tests.
The performance and accuracy are shown below. From this we see that WholeGraph has both higher performance and higher accuracy.

| FrameWork | Epoch time (s) | DGL Valid MRR| DGL Test MRR | OGB Valid MRR | OGB Test MRR |
|:---------:| :------------: | :----------: | :----------: | :-----------: | :----------: |
| DGL Pure GPU (1xA100)   | 1150 | 0.913  | 0.924  | 0.708 | 0.707 |
| DGL UVA (1xA100)    | 3125 | 0.909 | 0.929 | 0.698 | 0.696 |
| WholeGraph (1xA100) | 826 | 0.980 | 0.982 | 0.786 | 0.784 |
| WholeGraph (8xA100) | 114 | 0.994 | 0.970 | 0.792 | 0.793 |

## Heterogeneous Graph Example

Here we use [MAG240M-LSC](https://ogb.stanford.edu/kddcup2021/mag240m/) as an example:

### Dataset Preparation and Graph Construction

Same as node classification task, we need to download and convert the training data into WholeGraph's data format, and then build graph.
We use `examples/gnn/gnn_mag240m_data_process.py` to do these.

From `build` directory
```shell script
WHOLEGRAPH_PATH=...
export PYTHONPATH=${WHOLEGRAPH_PATH}/python:${WHOLEGRAPH_PATH}/build:${PYTHONPATH}
python ../example/gnn/gnn_mag240m_data_process.py -r PATH_TO_CONVERTED_DATA -p convert
python ../example/gnn/gnn_mag240m_data_process.py -r PATH_TO_CONVERTED_DATA -p build
```
`PATH_TO_CONVERTED_DATA` is the download path of OGB datasets, defaults to `dataset`, the downloaded data will be stored here.
And the converted and built data will be stored to `mag240m_kddcup2021/converted`.

### Training

After training data is downloaded, converted and graph built. Training can be simply done by `gnn_example_rgnn.py` script.

To run single node multi-GPU training:

We currently has optimized rGCN model in WholeGraph and rGraphSAGE and rGAT model with WholeMemory storage and DGL layers.

`-m` flag can be used to select model, valid values are `gcn`, `gat`, `sage`. `-f` flag can be used to select backend, valid values are `wg` and `dgl`.

```shell script
WHOLEGRAPH_PATH=...
export PYTHONPATH=${WHOLEGRAPH_PATH}/python:${WHOLEGRAPH_PATH}/build:${PYTHONPATH}
mpirun -np 8 python ${WHOLEGRAPH_PATH}/examples/gnn/gnn_example_rgnn.py -r PATH_TO_CONVERTED_DATA -m gat -f dgl
```

This script also support multi-node training, but need to set `MASTER_ADDR` and `MASTER_PORT` correctly.

### Performance

#### WholeMemory + DGL (rGAT model as an example)
Here we use the [MAG240M example of rGAT](https://github.com/dmlc/dgl/blob/0.8.2/examples/pytorch/ogb_lsc/MAG240M/train.py) from DGL as baseline.
Batchsize is 1024, sample count is [25,15], hidden size is 1024, and head number of GAT is 4.
The DGL version we use is 0.8.2.
All model configure and other hyper parameters are tried to be aligned with DGL's example.
The final script is [gnn_example_rgnn.py](../examples/gnn/gnn_example_rgnn.py).
`-f dgl` need to be specified to use dgl layers and `-m gat` need to be specified to use rGAT model. 

We train 10 epoch of DGL with single GPU as DGL's multi-GPU script seems don't work well.
All test run on DGX-A100 640G.
For WholeGraph, we use 8 GPUs.
The performance and accuracy are show below. From this we see that WholeGraph has higher performance and same accuracy.

| FrameWork | Epoch train time (s) | Valid Time (s) | Valid Accuracy @5th epoch | Valid Accuracy @40th epoch |
|:---------:| :------------: | :----------: | :----------: | :----------: |
| DGL (1xA100)   | 441 | 116  |  66.60  | - |
| WholeGraph (8xA100) | 11.2 | 0.8 | 66.88 | 68.89 |

#### WholeGraph (rGCN model)

Here we show the result of the optimized rGCN model in WholeGraph.
DGL also has rGCN implementation, we compare our optimized rGCN implementation with DGL's [RelGraphConv](https://github.com/dmlc/dgl/blob/0.8.2/python/dgl/nn/pytorch/conv/relgraphconv.py) layer.
All the underlying storage and sampler are using WholeGraph, and the GNN layers can use DGL or WholeGraph, which can be set by `-f` flag.
The DGL version we use is 0.8.2.
Batchsize is 1024, sample count is [25, 15] and [50, 50]. We tested hiddensize=400 and hiddensize=1024. Performance shown below. 
From this we see that WholeGraph has higher performance and same accuracy.

| FrameWork | Sample count | HiddenSize | Epoch train time (s) | Valid Time (s) | Valid Accuracy @40th epoch |
|:---------:|:------------:|:---------:| :------------: | :----------: | :----------: |
| WG + DGL (8xA100)   | [25, 15] | 400  | 4.42  | 0.41 | 67.82 |
| WholeGraph (8xA100) | [25, 15] | 400  | 1.13  | 0.12 | 67.76 |
| WG + DGL (8xA100)   | [50, 50] | 400  | 14.12 | 1.55 | 67.90 |
| WholeGraph (8xA100) | [50, 50] | 400  | 2.67  | 0.39 | 67.90 |
| WG + DGL (8xA100)   | [25, 15] | 1024 |  8.93 | 0.74 | 68.83 |
| WholeGraph (8xA100) | [25, 15] | 1024 | 1.52  | 0.14 | 68.85 |
| WG + DGL (8xA100)   | [50, 50] | 1024 | 28.19 | 2.76 | 69.01 |
| WholeGraph (8xA100) | [50, 50] | 1024 | 3.13  | 0.41 | 68.99 |

### Multi-node support

There are two kind of multi-node support in WholeGraph.
One is just to speed up training process. When the graph fits in one single server, and the features are not trainable, this applies, and we call this data parallel.
The other is to train larger graph, the embedding needs to be partitioned.

#### Multi-node support for data parallel

To train using data parallel, just run with multi-node without using NCCL Embedding.
We benchmark using ogbn-papers100M dataset with GCN model.
From the results, we see in this case, it is almost linear scaling.

| Node Count | # GPUs | Epoch Time | Speed Up |
|:----------:|:------:|:----------:|:--------:|
| 1 |  8 x A100 | 5.50 | 1.00 |
| 2 | 16 x A100 | 2.78 | 1.98 |
| 4 | 32 x A100 | 1.42 | 3.87 |
| 8 | 64 x A100 | 0.75 | 7.33 |

#### Multi-node support for large scale graph

For large GNN applications, graph structure of 1 billion nodes and 10 billion edges takes about 40 GB memory space.
However, feature vectors may be large and doesn't even fit in one single node.
For example 1 billion of 512 dimension node feature with type float takes about 2 TB memory.
So WholeMemory use WholeNCCLMemory to support large feature tables.

Here we use two large dataset: [ogbn-papers100M](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M) and [MAG240M-LSC](https://ogb.stanford.edu/kddcup2021/mag240m/) as examples.
The training scripts `gnn_example_node_classification.py` and `gnn_example_rgnn.py` also support multi-node training with node feature vector partitioning.
just add `--use_nccl` and the node features will be partitioned across all GPUs of all the machine nodes.
You may need to run multi-node MPI jobs, detailed launch commands depends on you cluster.

Here we show the performance of GraphSAGE model on ogbn-papers100M dataset and rGCN model on MAG240M-LSC.
For ogbn-papers100M, it is the same as the configure in Node Classification Task.
For MAG240M-LSC, the configuration is [25, 15] sample count and hiddensize is 1024.

| Exp. Setup | # GPUs | Papers100M Epoch Time (s) | MAG240M Epoch Time (s) |
|:----------:|:------:|:-------------------------:|:----------------------:|
| 1 node NVLink P2P |  8 x A100 | 5.54 | 1.52 |
| 1 node NCCL       |  8 x A100 | 7.02 | 1.58 |
| 2 nodes NCCL      | 16 x A100 | 7.78 | 1.19 |
| 4 nodes NCCL      | 32 x A100 | 5.89 | 0.75 |
| 8 nodes NCCL      | 64 x A100 | 3.67 | 0.46 |

For memory consumption, taking MAG240M for example, the converted graph has 3.46 billion edges, which takes about 14 GB memory.
And has 244 million nodes each with 768 dimension half feature, which takes about 366 GB memory.
So total memory consumption should be at least 380 GB.
We recorded the memory consumption during training process, which is shown below.
We see that use more GPUs reduce the memory needs of single GPU, which makes it possible to train larger graph.

| Exp. Setup | # GPUs | MAG240M Memory per GPU (GB) |
|:----------:|:------:|:---------------------------:|
| 1 node NVLink P2P |  8 x A100 |  61 GB  |
| 1 node NCCL       |  8 x A100 |  64 GB  |
| 2 nodes NCCL      | 16 x A100 |  39 GB  |
| 4 nodes NCCL      | 32 x A100 |  28 GB  |
| 8 nodes NCCL      | 64 x A100 |  20 GB  |
