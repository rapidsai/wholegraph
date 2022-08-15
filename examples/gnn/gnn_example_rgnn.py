# Copyright (c) 2022, NVIDIA CORPORATION.
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

import datetime
import os
import time
from optparse import OptionParser

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from apex.optimizers import FusedLAMB
from apex.parallel import DistributedDataParallel as DDP
from dgl.nn.pytorch import RelGraphConv
from mpi4py import MPI
from torch.utils.data import DataLoader
from wg_torch import comm as comm
from wg_torch import embedding_ops as embedding_ops
from wg_torch import graph_ops as graph_ops
from wg_torch.gnn.rGCNConv import RGCNConv as RGCNConv
from wg_torch.wm_tensor import *
from wg_torch.gnn.SAGEConv import SpmmMean

from wholegraph.torch import wholegraph_pytorch as wg

parser = OptionParser()
parser.add_option(
    "-r", "--root_dir", dest="root_dir", default="dataset", help="root graph directory."
)
parser.add_option(
    "-e", "--epochs", type="int", dest="epochs", default=80, help="number of epochs"
)
parser.add_option(
    "-b", "--batchsize", type="int", dest="batchsize", default=1024, help="batch size"
)
parser.add_option(
    "-c", "--classnum", type="int", dest="classnum", default=153, help="class number"
)
parser.add_option(
    "-n",
    "--neighbors",
    dest="neighbors",
    default="25,15",
    help="train neighboor sample count",
)
parser.add_option(
    "--hiddensize", type="int", dest="hiddensize", default=1024, help="hidden size"
)
parser.add_option(
    "-l", "--layernum", type="int", dest="layernum", default=2, help="layer number"
)
parser.add_option(
    "-f",
    "--framework",
    dest="framework",
    default="wg",
    help="framework type, valid values are: dgl, wg",
)
parser.add_option(
    "-m",
    "--model",
    dest="model",
    default="gcn",
    help="model type, valid values are: gcn, sage, gat",
)
parser.add_option(
    "-w",
    "--dataloaderworkers",
    type="int",
    dest="dataloaderworkers",
    default=0,
    help="number of workers for dataloader",
)
parser.add_option("--heads", type="int", dest="heads", default=4, help="num heads")
parser.add_option(
    "-d", "--dropout", type="float", dest="dropout", default=0.5, help="dropout"
)
parser.add_option(
    "-a",
    "--aggregator",
    dest="aggregator",
    default="sum",
    help="rGCN aggregator type, valid values are mean, sum",
)
parser.add_option("--lr", type="float", dest="lr", default=0.001, help="learning rate")
parser.add_option(
    "-t",
    "--truncate_dim",
    type="int",
    dest="truncate_dim",
    default=-1,
    help="class number",
)
parser.add_option(
    "--use_nccl",
    action="store_true",
    dest="use_nccl",
    default=False,
    help="whether use nccl for embeddings, default False",
)
(options, args) = parser.parse_args()

# https://github.com/snap-stanford/ogb/blob/master/examples/lsc/mag240m/rgnn.py
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb_lsc/MAG240M/train.py


use_chunked = True
use_host_memory = False


class DGLMAG240RGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_etypes,
        num_layers,
        num_heads,
        dropout,
        pred_ntype,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()

        if options.model == "gat":
            self.convs.append(
                torch.nn.ModuleList(
                    [
                        dglnn.GATConv(
                            in_channels,
                            hidden_channels // num_heads,
                            num_heads,
                            allow_zero_in_degree=True,
                        )
                        for _ in range(num_etypes)
                    ]
                )
            )
        else:
            assert options.model == "sage"
            self.convs.append(
                torch.nn.ModuleList(
                    [
                        dglnn.SAGEConv(in_channels, hidden_channels, "mean")
                        for _ in range(num_etypes)
                    ]
                )
            )
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.skips.append(torch.nn.Linear(in_channels, hidden_channels))
        if options.model == "gat":
            for _ in range(num_layers - 1):
                self.convs.append(
                    torch.nn.ModuleList(
                        [
                            dglnn.GATConv(
                                hidden_channels,
                                hidden_channels // num_heads,
                                num_heads,
                                allow_zero_in_degree=True,
                            )
                            for _ in range(num_etypes)
                        ]
                    )
                )
                self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
                self.skips.append(torch.nn.Linear(hidden_channels, hidden_channels))
        else:
            assert options.model == "sage"
            for _ in range(num_layers - 1):
                self.convs.append(
                    torch.nn.ModuleList(
                        [
                            dglnn.SAGEConv(hidden_channels, hidden_channels, "mean")
                            for _ in range(num_etypes)
                        ]
                    )
                )
                self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
                self.skips.append(torch.nn.Linear(hidden_channels, hidden_channels))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels),
        )
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_channels = hidden_channels
        self.pred_ntype = pred_ntype
        self.num_etypes = num_etypes

    def forward(self, mfgs, x):
        for i in range(len(mfgs)):
            mfg = mfgs[i]
            x_dst = x[: mfg.num_dst_nodes()]
            n_src = mfg.num_src_nodes()
            n_dst = mfg.num_dst_nodes()
            mfg = dgl.block_to_graph(mfg)
            x_skip = self.skips[i](x_dst)
            for j in range(self.num_etypes):
                subg = mfg.edge_subgraph(mfg.edata["etype"] == j, relabel_nodes=False)
                if subg.adj()._indices().shape[1] > 0:
                    x_skip += self.convs[i][j](subg, (x, x_dst)).view(
                        -1, self.hidden_channels
                    )
            x = self.norms[i](x_skip)
            x = F.elu(x)
            x = self.dropout(x)
        return self.mlp(x)


class DGLRGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_etypes,
        num_layers,
        num_heads,
        dropout,
        pred_ntype,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        assert options.model == "gcn"

        self.convs.append(RelGraphConv(in_channels, hidden_channels, num_etypes))
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                RelGraphConv(hidden_channels, hidden_channels, num_etypes)
            )
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels),
        )
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_channels = hidden_channels
        self.pred_ntype = pred_ntype
        self.num_etypes = num_etypes

    def forward(self, mfgs, x):
        for i in range(len(mfgs)):
            mfg = mfgs[i]
            mfg = dgl.block_to_graph(mfg)
            x = self.convs[i](mfg, x, mfg.edata["etype"])
            x = self.norms[i](x)
            x = F.elu(x)
            x = self.dropout(x)
        return self.mlp(x)


class MAG240HomoGraph(object):
    def __init__(self):
        super(MAG240HomoGraph, self).__init__()
        self.csr_row_ptr = None
        self.csr_col_ind = None
        self.csr_col_edge_type = None
        self.to_typed_id = None
        # paper, author, institution
        self.type_to_mixed_id = [None, None, None]
        self.features = None
        self.meta_dict = None

        self.paper_num = 0
        self.author_num = 0
        self.institution_num = 0
        self.feature_size = 768
        self.mixed_node_num = 0
        self.all_edges_count = 3456728464

        self.num_classes = 0

        self.edge_type_dict_cpu_array = None
        self.edge_type_dict_gpu_tensor = None
        self.edge_node_type_cpu_array = None
        self.edge_node_type_gpu_tensor = None
        self.num_ntypes = 3
        self.num_etypes = 5
        self.valid_edge_type_count = None

        self.homo_graph = graph_ops.HomoGraph()
        self.wm_comm = None

    def compute_features_nccl(self, target_node_type_id: int, target_edge_type_id: int):
        assert target_node_type_id == 1 or target_node_type_id == 2
        wm_comm = self.homo_graph.wm_nccl_embedding_comm
        torch.cuda.synchronize()
        wg.barrier(wm_comm)
        wm_rank = wg.get_rank(wm_comm)
        wm_size = wg.get_size(wm_comm)
        target_count = (
            self.author_num if target_node_type_id == 1 else self.institution_num
        )
        max_batch_size = 1024 * 4 if target_node_type_id == 1 else 32
        iter_count = (target_count + wm_size * max_batch_size - 1) // (
            wm_size * max_batch_size
        )
        for it in range(iter_count):
            torch.cuda.synchronize()
            wg.barrier(wm_comm)
            batch_start = min(it * max_batch_size * wm_size + wm_rank, target_count)
            batch_end = min((it + 1) * max_batch_size * wm_size + wm_rank, target_count)
            if batch_end == batch_start:
                batch_start, batch_end = wm_rank, wm_size + wm_rank
            batch_size = (batch_end - batch_start + wm_size - 1) // wm_size
            target_ids = torch.arange(
                batch_start,
                batch_start + batch_size * wm_size,
                wm_size,
                dtype=torch.int32,
                device="cuda",
            )
            mixed_ids = embedding_ops.embedding_lookup_nograd_common(
                self.type_to_mixed_id[target_node_type_id], target_ids
            )
            filter_target_value = torch.full(
                mixed_ids.shape,
                target_edge_type_id,
                device=mixed_ids.device,
                dtype=self.csr_col_edge_type.dtype,
            )
            (
                subgraph_csr_row_ptr,
                subgraph_csr_col_gid,
            ) = graph_ops.extract_subgraph_with_filter(
                graph_ops.GraphExtractType.EQUAL,
                mixed_ids,
                filter_target_value,
                self.homo_graph.edges_csr_row,
                self.homo_graph.edges_csr_col,
                self.csr_col_edge_type,
                False,
            )
            neighbor_features = embedding_ops.embedding_lookup_nograd_common(
                self.features, subgraph_csr_col_gid
            )
            subgraph_csr_col_idx = torch.arange(
                0, subgraph_csr_col_gid.shape[0], dtype=torch.int32, device="cuda"
            )
            sample_dup_count = torch.zeros([1])
            with torch.no_grad():
                target_features = SpmmMean.apply(
                    neighbor_features,
                    subgraph_csr_row_ptr.int(),
                    subgraph_csr_col_idx,
                    sample_dup_count,
                )
            embedding_ops.scatter_nograd(target_features, mixed_ids, self.features)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        wg.barrier(wm_comm)

    def load_and_compute_features_nccl(self, paper_embedding_filename):
        if comm.get_rank() == 0:
            print("load_and_compute_features_nccl...")
        wm_comm = self.homo_graph.wm_nccl_embedding_comm
        wm_tensor_type = WmTensorType.NCCL
        self.features = create_wm_tensor(
            wm_comm,
            [self.mixed_node_num, self.feature_size],
            [],
            torch.float16,
            wm_tensor_type,
        )
        wm_rank = wg.get_rank(wm_comm)
        wm_size = wg.get_size(wm_comm)
        feat_file_stat = os.stat(paper_embedding_filename)
        file_size = feat_file_stat.st_size
        assert file_size == 2 * self.feature_size * self.paper_num
        feat_start = wm_rank * self.paper_num // wm_size
        feat_end = (wm_rank + 1) * self.paper_num // wm_size
        max_batch_size = 1024 * 16
        iter_count = (self.paper_num + wm_size * max_batch_size - 1) // (
            wm_size * max_batch_size
        )
        for it in range(iter_count):
            batch_start = min(feat_start + it * max_batch_size, feat_end)
            batch_end = min(feat_start + (it + 1) * max_batch_size, feat_end)
            if batch_start == batch_end:
                batch_start, batch_end = 0, 1
            batch_size = batch_end - batch_start
            paper_ids = torch.arange(
                batch_start, batch_start + batch_size, dtype=torch.int32, device="cuda"
            )
            mixed_ids = embedding_ops.embedding_lookup_nograd_common(
                self.type_to_mixed_id[0], paper_ids
            )
            with open(paper_embedding_filename, "rb") as f:
                batch_feat_np = np.fromfile(
                    f,
                    dtype=np.float16,
                    count=batch_size * self.feature_size,
                    offset=batch_start * self.feature_size * 2,
                )
            batch_feat = (
                torch.from_numpy(batch_feat_np)
                .reshape([batch_size, self.feature_size])
                .cuda()
            )
            # print('batch_feat=%s\n' % (batch_feat, ))
            embedding_ops.scatter_nograd(batch_feat, mixed_ids, self.features)
        torch.cuda.synchronize()
        wg.barrier(wm_comm)
        if comm.get_rank() == 0:
            print("Load paper feature done")

        self.compute_features_nccl(1, 2)
        if comm.get_rank() == 0:
            print("Compute author feature done")

        self.compute_features_nccl(2, 4)
        if comm.get_rank() == 0:
            print("Compute institution feature done")

    def load_and_compute_features(self, paper_embedding_filename, wm_comm):
        global use_chunked
        global use_host_memory
        wm_tensor_type = get_intra_node_wm_tensor_type(use_chunked, use_host_memory)
        self.features = create_wm_tensor(
            wm_comm,
            [self.mixed_node_num, self.feature_size],
            [],
            torch.float16,
            wm_tensor_type,
        )
        wm_rank = wg.get_rank(wm_comm)
        wm_size = wg.get_size(wm_comm)

        feat_file_stat = os.stat(paper_embedding_filename)
        file_size = feat_file_stat.st_size
        assert file_size == 2 * self.feature_size * self.paper_num
        feat_start = wm_rank * self.paper_num // wm_size
        feat_end = (wm_rank + 1) * self.paper_num // wm_size
        max_batch_size = 1024 * 16

        for batch_start in range(feat_start, feat_end, max_batch_size):
            batch_size = min(max_batch_size, feat_end - batch_start)
            paper_ids = torch.arange(
                batch_start, batch_start + batch_size, dtype=torch.int32, device="cuda"
            )
            mixed_ids = embedding_ops.embedding_lookup_nograd_common(
                self.type_to_mixed_id[0], paper_ids
            )
            with open(paper_embedding_filename, "rb") as f:
                batch_feat_np = np.fromfile(
                    f,
                    dtype=np.float16,
                    count=batch_size * self.feature_size,
                    offset=batch_start * self.feature_size * 2,
                )
            batch_feat = (
                torch.from_numpy(batch_feat_np)
                .reshape([batch_size, self.feature_size])
                .cuda()
            )
            # print('batch_feat=%s\n' % (batch_feat, ))
            embedding_ops.scatter_nograd(batch_feat, mixed_ids, self.features)

        torch.cuda.synchronize()
        wg.barrier(wm_comm)
        if comm.get_rank() == 0:
            print("Load paper feature done")
        if use_chunked:
            wg.mixed_graph_sgc_chunked(
                self.features,
                self.csr_row_ptr,
                self.csr_col_ind,
                self.to_typed_id,
                1,
                0,
            )
        else:
            wg.mixed_graph_sgc(
                self.features,
                self.csr_row_ptr,
                self.csr_col_ind,
                self.to_typed_id,
                1,
                0,
            )
        torch.cuda.synchronize()
        wg.barrier(wm_comm)
        if comm.get_rank() == 0:
            print("Compute author feature done")
        if use_chunked:
            wg.mixed_graph_sgc_chunked(
                self.features,
                self.csr_row_ptr,
                self.csr_col_ind,
                self.to_typed_id,
                2,
                1,
            )
        else:
            wg.mixed_graph_sgc(
                self.features,
                self.csr_row_ptr,
                self.csr_col_ind,
                self.to_typed_id,
                2,
                1,
            )
        torch.cuda.synchronize()
        wg.barrier(wm_comm)
        if comm.get_rank() == 0:
            print("Compute institution feature done")

    def generate_csr_col_edge_type(self):
        global use_chunked
        global use_host_memory
        wm_tensor_type = get_intra_node_wm_tensor_type(use_chunked, use_host_memory)
        self.csr_col_edge_type = create_wm_tensor(
            self.homo_graph.wm_comm,
            [self.all_edges_count],
            [],
            self.edge_type_dict_gpu_tensor.dtype,
            wm_tensor_type,
        )
        wm_rank = wg.get_rank(self.homo_graph.wm_comm)
        wm_size = wg.get_size(self.homo_graph.wm_comm)
        node_start = wm_rank * self.mixed_node_num // wm_size
        node_end = (wm_rank + 1) * self.mixed_node_num // wm_size
        max_batch_size = 1024 * 16

        rank_edge_start = (
            embedding_ops.embedding_lookup_nograd_common(
                self.homo_graph.edges_csr_row,
                torch.full(
                    [1],
                    node_start,
                    dtype=self.homo_graph.edges_csr_col.dtype,
                    device="cuda",
                ),
            )
            .cpu()
            .item()
        )
        rank_edge_end = (
            embedding_ops.embedding_lookup_nograd_common(
                self.homo_graph.edges_csr_row,
                torch.full(
                    [1],
                    node_end,
                    dtype=self.homo_graph.edges_csr_col.dtype,
                    device="cuda",
                ),
            )
            .cpu()
            .item()
        )
        rank_edge_idx = rank_edge_start
        for batch_start in range(node_start, node_end, max_batch_size):
            batch_size = min(max_batch_size, node_end - batch_start)
            mixed_ids = torch.arange(
                batch_start,
                batch_start + batch_size,
                dtype=self.csr_col_ind.dtype,
                device="cuda",
            )
            (
                subgraph_csr_row,
                subgraph_csr_col_gid,
                _,
            ) = graph_ops.unweighted_sample_without_replacement_single_layer(
                mixed_ids,
                self.homo_graph.edges_csr_row,
                self.homo_graph.edges_csr_col,
                -1,
            )
            edge_type_tensor = self.get_edge_type(
                mixed_ids, subgraph_csr_row, subgraph_csr_col_gid
            )
            batch_edge_count = subgraph_csr_col_gid.shape[0]
            edge_indice = torch.arange(
                rank_edge_idx,
                rank_edge_idx + batch_edge_count,
                dtype=torch.int64,
                device="cuda",
            )
            embedding_ops.scatter_nograd(
                edge_type_tensor, edge_indice, self.csr_col_edge_type
            )
            rank_edge_idx = rank_edge_idx + batch_edge_count
        assert rank_edge_idx == rank_edge_end

        wg.barrier(self.homo_graph.wm_comm)

    def load(self, wm_comm, wm_nccl_embedding_comm=None):
        self.homo_graph.wm_comm = wm_comm
        self.homo_graph.wm_nccl_embedding_comm = wm_nccl_embedding_comm
        self.wm_comm = wm_comm
        dir_name = "mag240m_kddcup2021"
        output_dir = os.path.join(options.root_dir, dir_name, "converted")
        meta_file_name = "meta.yaml"
        with open(os.path.join(output_dir, meta_file_name), "r", encoding="utf-8") as f:
            self.meta_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.num_classes = self.meta_dict["num_classes"]
        self.paper_num = self.meta_dict["nodes"]["paper"]["num"]
        self.author_num = self.meta_dict["nodes"]["author"]["num"]
        self.institution_num = self.meta_dict["nodes"]["institution"]["num"]
        assert self.feature_size == self.meta_dict["nodes"]["paper"]["feature_dim"]
        self.mixed_node_num = self.paper_num + self.author_num + self.institution_num

        global use_chunked
        global use_host_memory
        wm_tensor_type = get_intra_node_wm_tensor_type(use_chunked, use_host_memory)
        self.csr_row_ptr = create_wm_tensor_from_file(
            [self.mixed_node_num + 1],
            torch.int64,
            wm_comm,
            os.path.join(output_dir, "mixed_graph_csr_row_ptr"),
            wm_tensor_type,
        )
        self.csr_col_ind = create_wm_tensor_from_file(
            [self.all_edges_count],
            torch.int32,
            wm_comm,
            os.path.join(output_dir, "mixed_graph_csr_col_idx"),
            wm_tensor_type,
        )
        self.to_typed_id = create_wm_tensor_from_file(
            [self.mixed_node_num],
            torch.int64,
            wm_comm,
            os.path.join(output_dir, "mixed_graph_id_mapping_mixed_to_typed"),
            wm_tensor_type,
        )
        self.type_to_mixed_id[0] = create_wm_tensor_from_file(
            [self.paper_num],
            torch.int32,
            wm_comm,
            os.path.join(output_dir, "mixed_graph_id_mapping_paper"),
            wm_tensor_type,
        )
        self.type_to_mixed_id[1] = create_wm_tensor_from_file(
            [self.author_num],
            torch.int32,
            wm_comm,
            os.path.join(output_dir, "mixed_graph_id_mapping_author"),
            wm_tensor_type,
        )
        with open(
            os.path.join(output_dir, "mixed_graph_id_mapping_institution"), "rb"
        ) as f:
            institution_mapping_np_array = np.fromfile(f, dtype=np.int32)
        self.type_to_mixed_id[2] = torch.from_numpy(institution_mapping_np_array).cuda()

        self.edge_type_dict_cpu_array = [[0, 1, -1], [2, -1, 3], [-1, 4, -1]]
        self.edge_type_dict_gpu_tensor = torch.CharTensor(
            self.edge_type_dict_cpu_array, device="cpu"
        ).cuda()
        self.edge_node_type_cpu_array = [[0, 0], [0, 1], [1, 0], [1, 2], [2, 1]]
        self.edge_node_type_gpu_tensor = torch.CharTensor(
            self.edge_node_type_cpu_array, device="cpu"
        ).cuda()
        self.valid_edge_type_count = [2, 4, 5, 5, 5, 5, 5]

        self.homo_graph.edges_csr_row = self.csr_row_ptr
        self.homo_graph.edges_csr_col = self.csr_col_ind
        self.homo_graph.node_count = self.mixed_node_num
        self.homo_graph.edge_count = self.all_edges_count
        self.homo_graph.is_chunked = use_chunked
        self.homo_graph.use_host_memory = use_host_memory

        self.generate_csr_col_edge_type()

        if self.homo_graph.wm_nccl_embedding_comm is None:
            self.load_and_compute_features(
                os.path.join(output_dir, "node_feat____paper.bin"), self.wm_comm
            )
        else:
            self.load_and_compute_features_nccl(
                os.path.join(output_dir, "node_feat____paper.bin")
            )
        self.homo_graph.node_feat = self.features

    def verify_features(self):
        """
        verifies features by DGL results, this helps confirm that the features and graph structures are correct.
        :return:
        """
        dir_name = "mag240m_kddcup2021"
        dgl_data_dir = os.path.join(options.root_dir, dir_name, "dgl_data")
        wm_rank = wg.get_rank(self.wm_comm)
        wm_size = wg.get_size(self.wm_comm)

        dgl_author_embedding_filename = os.path.join(dgl_data_dir, "author.npy")

        feat_file_stat = os.stat(dgl_author_embedding_filename)
        file_size = feat_file_stat.st_size
        assert file_size == 2 * self.feature_size * self.author_num
        feat_start = wm_rank * self.author_num // wm_size
        feat_end = (wm_rank + 1) * self.author_num // wm_size
        max_batch_size = 1024 * 16

        for batch_start in range(feat_start, feat_end, max_batch_size):
            batch_size = min(max_batch_size, feat_end - batch_start)
            author_ids = torch.arange(
                batch_start, batch_start + batch_size, dtype=torch.int32, device="cuda"
            )
            mixed_ids = embedding_ops.embedding_lookup_nograd_common(
                self.type_to_mixed_id[1], author_ids
            )
            with open(dgl_author_embedding_filename, "rb") as f:
                batch_feat_np = np.fromfile(
                    f,
                    dtype=np.float16,
                    count=batch_size * self.feature_size,
                    offset=batch_start * self.feature_size * 2,
                )
            batch_feat = (
                torch.from_numpy(batch_feat_np)
                .reshape([batch_size, self.feature_size])
                .cuda()
            )
            computed_author_feat = embedding_ops.embedding_lookup_nograd_common(
                self.features, mixed_ids
            )
            if not torch.allclose(
                batch_feat, computed_author_feat, rtol=1e-2, atol=1e-3
            ):
                print("author dgl=%s\nwg=%s\n" % (batch_feat, computed_author_feat))
                assert False
        torch.cuda.synchronize()
        wg.barrier(self.wm_comm)
        print("Author All Close!")

        dgl_institution_embedding_filename = os.path.join(dgl_data_dir, "inst.npy")

        feat_file_stat = os.stat(dgl_institution_embedding_filename)
        file_size = feat_file_stat.st_size
        assert file_size == 2 * self.feature_size * self.institution_num
        feat_start = wm_rank * self.institution_num // wm_size
        feat_end = (wm_rank + 1) * self.institution_num // wm_size
        max_batch_size = 1024 * 16

        for batch_start in range(feat_start, feat_end, max_batch_size):
            batch_size = min(max_batch_size, feat_end - batch_start)
            institution_ids = torch.arange(
                batch_start, batch_start + batch_size, dtype=torch.int32, device="cuda"
            )
            mixed_ids = embedding_ops.embedding_lookup_nograd_common(
                self.type_to_mixed_id[2], institution_ids
            )
            with open(dgl_institution_embedding_filename, "rb") as f:
                batch_feat_np = np.fromfile(
                    f,
                    dtype=np.float16,
                    count=batch_size * self.feature_size,
                    offset=batch_start * self.feature_size * 2,
                )
            batch_feat = (
                torch.from_numpy(batch_feat_np)
                .reshape([batch_size, self.feature_size])
                .cuda()
            )
            computed_institution_feat = embedding_ops.embedding_lookup_nograd_common(
                self.features, mixed_ids
            )
            if not torch.allclose(
                batch_feat, computed_institution_feat, rtol=1e-2, atol=1e-3
            ):
                print(
                    "institution dgl=%s\nwg=%s\n"
                    % (batch_feat, computed_institution_feat)
                )
                assert False
        torch.cuda.synchronize()
        wg.barrier(self.wm_comm)
        print("Institution All Close!")

    def get_edge_type(self, src_mixid, sub_graph_csr_row_ptr, sub_graph_csr_col_mixid):
        global use_chunked
        if use_chunked:
            return torch.ops.wholegraph.get_csr_mixed_sub_graph_edge_types_chunked(
                src_mixid,
                sub_graph_csr_row_ptr,
                sub_graph_csr_col_mixid,
                self.edge_type_dict_gpu_tensor,
                self.to_typed_id.get_ptr(),
                self.num_ntypes,
                self.num_etypes,
            )
        else:
            return torch.ops.wholegraph.get_csr_mixed_sub_graph_edge_types(
                src_mixid,
                sub_graph_csr_row_ptr,
                sub_graph_csr_col_mixid,
                self.edge_type_dict_gpu_tensor,
                self.to_typed_id,
                self.num_ntypes,
                self.num_etypes,
            )

    def truncate_dim(self):
        if options.truncate_dim == -1 or options.truncate_dim == self.feature_size:
            options.truncate_dim = self.feature_size
            return
        assert 0 < options.truncate_dim <= self.feature_size
        if isinstance(self.homo_graph.node_feat, torch.Tensor):
            self.homo_graph.node_feat = self.homo_graph.node_feat[
                :, : options.truncate_dim
            ]
        else:
            self.homo_graph.node_feat = wg.get_sub_chunked_tensor(
                self.homo_graph.node_feat, [0, 0], [-1, 64]
            )
        self.feature_size = options.truncate_dim


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


def build_dgl_subgraph(idx: torch.Tensor, graph: MAG240HomoGraph, max_neighbors):
    paper_ids = idx.cuda()
    mixed_paper_ids = embedding_ops.embedding_lookup_nograd_common(
        graph.type_to_mixed_id[0], paper_ids
    )
    (
        target_gids,
        edge_indice,
        csr_row_ptrs,
        csr_col_inds,
        sample_dup_counts,
    ) = graph.homo_graph.unweighted_sample_without_replacement(
        mixed_paper_ids, max_neighbors
    )
    mfgs = []
    for layer in range(options.layernum):
        block = dgl.create_block(
            ("csc", (csr_row_ptrs[layer], csr_col_inds[layer], torch.IntTensor([]))),
            device="cuda",
        )
        edge_type = graph.get_edge_type(
            target_gids[layer + 1],
            csr_row_ptrs[layer],
            target_gids[layer][csr_col_inds[layer].long()],
        )
        block.edata["etype"] = edge_type
        mfgs += [block]
    x = graph.homo_graph.gather(target_gids[0], dtype=torch.float32)
    return mfgs, x


class WGRGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_etypes,
        num_layers,
        num_heads,
        dropout,
        pred_ntype,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        assert options.model == "gcn"

        self.convs.append(
            RGCNConv(
                in_channels, hidden_channels, num_etypes, True, True, options.aggregator
            )
        )
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_etypes,
                    True,
                    True,
                    options.aggregator,
                )
            )
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels),
        )
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_channels = hidden_channels
        self.pred_ntype = pred_ntype
        self.num_etypes = num_etypes

    def forward(self, subgs, x):
        for i in range(len(subgs)):
            subg = subgs[i]
            x = self.convs[i](subg, x)
            x = self.norms[i](x)
            x = F.elu(x)
            x = self.dropout(x)
        return self.mlp(x)


"""
def bucket_mixed_ids(mixed_ids: torch.Tensor, mixed_ids_to_typed_table: torch.Tensor, node_type_count: int,
                     guarantee_ntype: Union[int, None] = None):
    if guarantee_ntype is not None:
        # don't change order if guaranteed same node count
        bucket_csr_array = [0 if i <= guarantee_ntype else mixed_ids.shape[0] for i in range(node_type_count + 1)]
        return mixed_ids, torch.IntTensor(bucket_csr_array, device='cpu'), None
    typed_ids = embedding_ops.embedding_lookup_nograd_common(mixed_ids_to_typed_table, mixed_ids)
    sorted_typed_ids, raw_indice = torch.sort(typed_ids)
    bucketed_mixed_ids = mixed_ids[raw_indice]
    bucket_csr_tensor = torch.ops.wholegraph.get_bucketed_csr_from_sorted_typed_ids(sorted_typed_ids,
                                                                                     node_type_count).cpu()
    return bucketed_mixed_ids, bucket_csr_tensor, raw_indice
"""


def build_wg_subgraph(idx: torch.Tensor, graph: MAG240HomoGraph, max_neighbors):
    paper_ids = idx.cuda()
    mixed_paper_ids = embedding_ops.embedding_lookup_nograd_common(
        graph.type_to_mixed_id[0], paper_ids
    )
    hops = len(max_neighbors)
    target_gids = [None] * (hops + 1)
    target_gids[hops] = mixed_paper_ids
    sub_graphs = [None] * hops
    for i in range(hops - 1, -1, -1):
        (
            neighboor_gids_offset,
            neighboor_gids_vdata,
            neighboor_src_lids,
        ) = graph_ops.unweighted_sample_without_replacement_single_layer(
            target_gids[i + 1],
            graph.csr_row_ptr,
            graph.csr_col_ind,
            max_neighbors[hops - i - 1],
        )
        edge_type = graph.get_edge_type(
            target_gids[i + 1], neighboor_gids_offset, neighboor_gids_vdata
        )
        edge_typed_ids = torch.ops.wholegraph.pack_to_typed_ids(
            neighboor_gids_vdata, edge_type
        )
        bucketed_unique_ids_packed, raw_to_unique_mapping, dup_count = torch.unique(
            edge_typed_ids, return_inverse=True, return_counts=True
        )
        edge_bucket_csr_tensor = (
            torch.ops.wholegraph.get_bucketed_csr_from_sorted_typed_ids(
                bucketed_unique_ids_packed, graph.num_etypes
            )
        )
        bucketed_unique_ids, _ = torch.ops.wholegraph.unpack_typed_ids(
            bucketed_unique_ids_packed
        )
        edge_bucket_csr = edge_bucket_csr_tensor.cpu().detach().numpy().tolist()
        num_valid_relation = graph.valid_edge_type_count[hops - 1 - i]
        sub_graphs[i] = {
            "target_ids": target_gids[i + 1],
            "unique_neighbor_ids": bucketed_unique_ids,
            "edge_bucket_csr": edge_bucket_csr,
            "edge_type": edge_type,
            "csr_row_ptr": neighboor_gids_offset,
            "csr_col_ind": raw_to_unique_mapping.int(),
            "dup_count": dup_count.int(),
            "num_relation": num_valid_relation,
        }
        target_gids[i] = torch.cat([target_gids[i + 1], bucketed_unique_ids])
    x = graph.homo_graph.gather(target_gids[0], dtype=torch.float32)
    return sub_graphs, x


def build_subgraph(idx: torch.Tensor, graph: MAG240HomoGraph, max_neighbors):
    assert options.framework == "dgl" or options.framework == "wg"
    if options.framework == "dgl":
        return build_dgl_subgraph(idx, graph, max_neighbors)
    elif options.framework == "wg":
        return build_wg_subgraph(idx, graph, max_neighbors)
    else:
        return None


def train(train_data, valid_data, model, graph, optimizer):
    print("start training...")
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
    train_dataset = graph_ops.NodeClassificationDataset(
        train_data, comm.get_rank(), comm.get_world_size()
    )
    valid_dataset = graph_ops.NodeClassificationDataset(
        valid_data, comm.get_rank(), comm.get_world_size()
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=comm.get_world_size(),
        rank=comm.get_rank(),
        shuffle=True,
        drop_last=False,
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,
        num_replicas=comm.get_world_size(),
        rank=comm.get_rank(),
        shuffle=False,
        drop_last=False,
    )

    max_neighbors = parse_max_neighbors(options.layernum, options.neighbors)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=options.batchsize,
        num_workers=options.dataloaderworkers,
        pin_memory=True,
        sampler=train_sampler,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=options.batchsize,
        num_workers=options.dataloaderworkers,
        pin_memory=True,
        sampler=valid_sampler,
    )
    print(
        "[%s] Starting train loop."
        % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),)
    )
    step_id = 0
    for epoch in range(options.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        for i, (idx, label) in enumerate(train_dataloader):
            y = torch.reshape(label, (-1,)).cuda()
            mfgs, x = build_subgraph(idx, graph, max_neighbors)
            y_hat = model(mfgs, x)
            loss = F.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (y_hat.argmax(1) == y).float().mean()
            if step_id % 100 == 0 and comm.get_rank() == 0:
                print(
                    "[%s] step=%d, loss=%f, acc=%f"
                    % (
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        step_id,
                        loss.cpu().item(),
                        acc.cpu().item(),
                    )
                )
            step_id += 1

        epoch_train_end = time.time()

        if comm.get_rank() == 0:
            print(
                "[%s] start valid train_time=%f"
                % (
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch_train_end - epoch_start,
                )
            )
        model.eval()
        correct = torch.LongTensor([0]).cuda()
        total = torch.LongTensor([0]).cuda()
        for i, (idx, label) in enumerate(valid_dataloader):
            with torch.no_grad():
                y = torch.reshape(label, (-1,)).cuda()
                mfgs, x = build_subgraph(idx, graph, max_neighbors)
                y_hat = model(mfgs, x)
                correct += (y_hat.argmax(1) == y).sum().item()
                total += y_hat.shape[0]

        # `reduce` data into process 0
        torch.distributed.reduce(correct, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(total, dst=0, op=torch.distributed.ReduceOp.SUM)
        acc = (correct / total).cpu().item()
        epoch_valid_end = time.time()
        if comm.get_rank() == 0:
            print(
                "[%s] [VALID] time=%f epoch=%d acc=%f"
                % (
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch_valid_end - epoch_train_end,
                    epoch,
                    acc,
                )
            )

        sched.step()


def main():
    print(
        "[%s] Starting process."
        % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),)
    )
    wg.init_lib()
    torch.set_num_threads(1)
    comma = MPI.COMM_WORLD
    shared_comma = comma.Split_type(MPI.COMM_TYPE_SHARED)
    os.environ["RANK"] = str(comma.Get_rank())
    os.environ["WORLD_SIZE"] = str(comma.Get_size())
    # slurm in Selene has MASTER_ADDR env
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12335"
    local_rank = shared_comma.Get_rank()
    local_size = shared_comma.Get_size()
    assert options.model == "sage" or options.model == "gat" or options.model == "gcn"
    # print("[%s] Rank=%d, local_rank=%d" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), local_rank, comma.Get_rank()))
    dev_count = torch.cuda.device_count()
    assert dev_count > 0
    assert local_size <= dev_count
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    wm_comm = create_intra_node_communicator(
        comma.Get_rank(), comma.Get_size(), local_size
    )
    wm_embedding_comm = None
    if options.use_nccl:
        if comma.Get_rank() == 0:
            print("Using nccl embeddings.")
        wm_embedding_comm = create_global_communicator(
            comma.Get_rank(), comma.Get_size()
        )

    dir_name = "mag240m_kddcup2021"
    data_dir = os.path.join(options.root_dir, dir_name, "converted")
    train_data, valid_data, test_data = graph_ops.load_pickle_data(data_dir, "mag240m")

    graph = MAG240HomoGraph()
    graph.load(wm_comm, wm_embedding_comm)
    graph.truncate_dim()

    # graph.verify_features()

    print("Rank=%d, Graph loaded." % (comma.Get_rank(),))
    if options.framework == "dgl":
        if options.model == "sage" or options.model == "gat":
            model = DGLMAG240RGNN(
                graph.feature_size,
                graph.num_classes,
                options.hiddensize,
                graph.num_etypes,
                options.layernum,
                options.heads,
                options.dropout,
                "paper",
            ).cuda()
        else:
            model = DGLRGCN(
                graph.feature_size,
                graph.num_classes,
                options.hiddensize,
                graph.num_etypes,
                options.layernum,
                options.heads,
                options.dropout,
                "paper",
            ).cuda()
    elif options.framework == "wg":
        model = WGRGCN(
            graph.feature_size,
            graph.num_classes,
            options.hiddensize,
            graph.num_etypes,
            options.layernum,
            options.heads,
            options.dropout,
            "paper",
        ).cuda()
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print("Rank=%d, model created." % (comma.Get_rank(),))
    # model.cuda()
    print("Rank=%d, model moved to cuda." % (comma.Get_rank(),))
    model = DDP(model, delay_allreduce=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    optimizer = FusedLAMB(model.parameters(), lr=options.lr, eps=1e-8, weight_decay=0)
    # optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=options.lr)
    print("Rank=%d, optimizer created." % (comma.Get_rank(),))

    train(train_data, valid_data, model, graph, optimizer)
    # if comm.get_rank() == 0:
    #    test(test_data, model)

    wg.finalize_lib()
    print("Rank=%d, wholegraph shutdown." % (comma.Get_rank(),))


if __name__ == "__main__":
    main()
