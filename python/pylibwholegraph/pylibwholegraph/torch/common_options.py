# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser


def add_training_options(argparser: ArgumentParser):
    argparser.add_argument(
        "-e", "--epochs", type=int, dest="epochs", default=24, help="number of epochs"
    )
    argparser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        dest="batchsize",
        default=1024,
        help="batch size",
    )
    argparser.add_argument(
        "--lr", type=float, dest="lr", default=0.003, help="learning rate"
    )
    argparser.add_argument(
        "--embedding-memory-type",
        dest="embedding_memory_type",
        default="chunked",
        help="Embedding memory type, should be: continuous, chunked, distributed, hierarchy",
    )
    argparser.add_argument(
        "--cache-type",
        dest="cache_type",
        default="none",
        help="Embedding cache type, should be: none, local_device, local_node or all_devices",
    )
    argparser.add_argument(
        "--cache-ratio",
        type=float,
        dest="cache_ratio",
        default=0.5,
        help="cache ratio",
    )
    argparser.add_argument(
        "--use-cpp-ext",
        action="store_true",
        dest="use_cpp_ext",
        default=False,
        help="Whether to use cpp extension for pytorch"
    )
    argparser.add_argument(
        "--train-embedding",
        action="store_true",
        dest="train_embedding",
        default=False,
        help="Whether to train embedding",
    )
    argparser.add_argument(
        "--distributed-backend-type",
        dest="distributed_backend_type",
        default="nccl",
        help="distributed backend type, should be: nccl, nvshmem ",
    )
    argparser.add_argument(
        "--log-level",
        dest="log_level",
        default="info",
        help="Logging level of wholegraph, should be: trace, debug, info, warn, error"
    )


def add_common_graph_options(argparser: ArgumentParser):
    argparser.add_argument(
        "-r",
        "--root-dir",
        dest="root_dir",
        default="dataset",
        help="graph dataset root directory.",
    )
    argparser.add_argument(
        "--use-global-embedding",
        action="store_true",
        dest="use_global_embedding",
        default=False,
        help="Store embedding across all ranks or only in local node.",
    )
    argparser.add_argument(
        "--feat-dim",
        type=int,
        dest="feat_dim",
        default=100,
        help="default feature dim",
    )
    argparser.add_argument(
        "--round-robin-size",
        type=int,
        dest="round_robin_size",
        default=0,
        help="continuous embedding number for each rank whiling using round-robin sharding strategy, \
                0 for not using round-robin shard strategy",
    )


def add_common_model_options(argparser: ArgumentParser):
    argparser.add_argument(
        "--hiddensize", type=int, dest="hiddensize", default=256, help="hidden size"
    )
    argparser.add_argument(
        "-l", "--layernum", type=int, dest="layernum", default=3, help="layer number"
    )
    argparser.add_argument(
        "-m",
        "--model",
        dest="model",
        default="sage",
        help="model type, valid values are: sage, gcn, gat",
    )
    argparser.add_argument(
        "-f",
        "--framework",
        dest="framework",
        default="cugraph",
        help="framework type, valid values are: dgl, pyg, wg, cugraph",
    )
    argparser.add_argument("--heads", type=int, dest="heads", default=4, help="num heads")
    argparser.add_argument(
        "-d", "--dropout", type=float, dest="dropout", default=0.5, help="dropout"
    )


def add_common_sampler_options(argparser: ArgumentParser):
    argparser.add_argument(
        "-n",
        "--neighbors",
        dest="neighbors",
        default="30,30,30",
        help="train neighboor sample count",
    )
    argparser.add_argument(
        "-s",
        "--inferencesample",
        type=str,
        dest="inferencesample",
        default="30",
        help="inference sample count, -1 is all",
    )


def add_node_classfication_options(argparser: ArgumentParser):
    argparser.add_argument(
        "-c",
        "--classnum",
        type=int,
        dest="classnum",
        default=172,
        help="class number",
    )


def add_dataloader_options(argparser: ArgumentParser):
    argparser.add_argument(
        "--pickle-data-path",
        dest="pickle_data_path",
        default="",
        help="training data file path, should be pickled dict",
    )
    argparser.add_argument(
        "-w",
        "--dataloaderworkers",
        type=int,
        dest="dataloaderworkers",
        default=0,
        help="number of workers for dataloader",
    )


def parse_max_neighbors(num_layer, neighbor_str):
    neighbor_str_vec = neighbor_str.split(",")
    max_neighbors = []
    for ns in neighbor_str_vec:
        max_neighbors.append(int(ns))
    assert len(max_neighbors) == 1 or len(max_neighbors) == num_layer
    if len(max_neighbors) != num_layer:
        for i in range(1, num_layer):
            max_neighbors.append(max_neighbors[0])
    # max_neighbors.reverse()
    return max_neighbors
