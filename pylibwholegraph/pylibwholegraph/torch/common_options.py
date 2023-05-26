# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from optparse import OptionParser


def add_training_options(parser: OptionParser):
    parser.add_option(
        "-e", "--epochs", type="int", dest="epochs", default=24, help="number of epochs"
    )
    parser.add_option(
        "-b",
        "--batchsize",
        type="int",
        dest="batchsize",
        default=1024,
        help="batch size",
    )
    parser.add_option(
        "--lr", type="float", dest="lr", default=0.003, help="learning rate"
    )
    parser.add_option(
        "--embedding-memory-type",
        dest="embedding_memory_type",
        default="chunked",
        help="Embedding memory type, should be: continuous, chunked or distributed",
    )
    parser.add_option(
        "--cache-type",
        dest="cache_type",
        default="none",
        help="Embedding cache type, should be: none, local_device, local_node or all_devices",
    )
    parser.add_option(
        "--cache-ratio",
        type="float",
        dest="cache_ratio",
        default=0.5,
        help="cache ratio",
    )
    parser.add_option(
        "--train-embedding",
        action="store_true",
        dest="train_embedding",
        default=False,
        help="Whether to train embedding",
    )


def add_common_graph_options(parser: OptionParser):
    parser.add_option(
        "-r",
        "--root-dir",
        dest="root_dir",
        default="dataset",
        help="graph dataset root directory.",
    )
    parser.add_option(
        "--use-global-embedding",
        action="store_true",
        dest="use_global_embedding",
        default=False,
        help="Store embedding across all ranks or only in local node.",
    )
    parser.add_option(
        "--feat-dim",
        type="int",
        dest="feat_dim",
        default=100,
        help="default feature dim",
    )


def add_common_model_options(parser: OptionParser):
    parser.add_option(
        "--hiddensize", type="int", dest="hiddensize", default=256, help="hidden size"
    )
    parser.add_option(
        "-l", "--layernum", type="int", dest="layernum", default=3, help="layer number"
    )
    parser.add_option(
        "-m",
        "--model",
        dest="model",
        default="sage",
        help="model type, valid values are: sage, gcn, gat",
    )
    parser.add_option(
        "-f",
        "--framework",
        dest="framework",
        default="cugraph",
        help="framework type, valid values are: dgl, pyg, wg, cugraph",
    )
    parser.add_option("--heads", type="int", dest="heads", default=1, help="num heads")
    parser.add_option(
        "-d", "--dropout", type="float", dest="dropout", default=0.5, help="dropout"
    )


def add_common_sampler_options(parser: OptionParser):
    parser.add_option(
        "-n",
        "--neighbors",
        dest="neighbors",
        default="30,30,30",
        help="train neighboor sample count",
    )
    parser.add_option(
        "-s",
        "--inferencesample",
        type="int",
        dest="inferencesample",
        default=30,
        help="inference sample count, -1 is all",
    )


def add_node_classfication_options(parser: OptionParser):
    parser.add_option(
        "-c",
        "--classnum",
        type="int",
        dest="classnum",
        default=172,
        help="class number",
    )


def add_dataloader_options(parser: OptionParser):
    parser.add_option(
        "--pickle-data-path",
        dest="pickle_data_path",
        default="",
        help="training data file path, should be pickled dict",
    )
    parser.add_option(
        "-w",
        "--dataloaderworkers",
        type="int",
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
