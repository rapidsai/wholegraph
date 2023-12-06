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

import torch
from .graph_structure import GraphStructure
from .embedding import WholeMemoryEmbedding, WholeMemoryEmbeddingModule
from .common_options import parse_max_neighbors
import torch.nn.functional as F
from .graph_ops import add_csr_self_loop


framework_name = None


def set_framework(framework: str):
    global framework_name
    assert framework_name is None
    framework_name = framework
    global SAGEConv, GATConv
    if framework_name == "dgl":
        global dgl
        import dgl
        from dgl.nn.pytorch.conv import SAGEConv, GATConv
    elif framework_name == "pyg":
        global SparseTensor
        from torch_sparse import SparseTensor
        from torch_geometric.nn import SAGEConv, GATConv
    elif framework_name == "wg":
        from wg_torch.gnn.SAGEConv import SAGEConv
        from wg_torch.gnn.GATConv import GATConv
    elif framework_name == "cugraph":
        from .cugraphops.sage_conv import CuGraphSAGEConv as SAGEConv
        from .cugraphops.gat_conv import CuGraphGATConv as GATConv


def create_gnn_layers(
    in_feat_dim, hidden_feat_dim, class_count, num_layer, num_head, model_type
):
    gnn_layers = torch.nn.ModuleList()
    global framework_name
    for i in range(num_layer):
        layer_output_dim = (
            hidden_feat_dim // num_head if i != num_layer - 1 else class_count
        )
        layer_input_dim = in_feat_dim if i == 0 else hidden_feat_dim
        mean_output = True if i == num_layer - 1 else False
        if framework_name == "pyg":
            if model_type == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            elif model_type == "gat":
                concat = not mean_output
                gnn_layers.append(
                    GATConv(
                        layer_input_dim, layer_output_dim, heads=num_head, concat=concat
                    )
                )
            else:
                assert model_type == "gcn"
                gnn_layers.append(
                    SAGEConv(layer_input_dim, layer_output_dim, root_weight=False)
                )
        elif framework_name == "dgl":
            if model_type == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, "mean"))
            elif model_type == "gat":
                gnn_layers.append(
                    GATConv(
                        layer_input_dim,
                        layer_output_dim,
                        num_heads=num_head,
                        allow_zero_in_degree=True,
                    )
                )
            else:
                assert model_type == "gcn"
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, "gcn"))
        elif framework_name == "wg":
            if model_type == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            elif model_type == "gat":
                gnn_layers.append(
                    GATConv(
                        layer_input_dim,
                        layer_output_dim,
                        num_heads=num_head,
                        mean_output=mean_output,
                    )
                )
            else:
                assert model_type == "gcn"
                gnn_layers.append(
                    SAGEConv(layer_input_dim, layer_output_dim, aggregator="gcn")
                )
        elif framework_name == "cugraph":
            assert model_type == "sage" or model_type == "gat"
            if model_type == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            elif model_type == "gat":
                concat = not mean_output
                gnn_layers.append(
                    GATConv(
                        layer_input_dim, layer_output_dim, heads=num_head, concat=concat
                    )
                )
    return gnn_layers


def create_sub_graph(
    target_gid,
    target_gid_1,
    edge_data,
    csr_row_ptr,
    csr_col_ind,
    max_num_neighbors: int,
    add_self_loop: bool,
):
    global framework_name
    if framework_name == "pyg":
        neighboor_dst_unique_ids = csr_col_ind
        neighboor_src_unique_ids = edge_data[1]
        target_neighbor_count = target_gid.size()[0]
        if add_self_loop:
            self_loop_ids = torch.arange(
                0,
                target_gid_1.size()[0],
                dtype=neighboor_dst_unique_ids.dtype,
                device=target_gid.device,
            )
            edge_index = SparseTensor(
                row=torch.cat([neighboor_src_unique_ids, self_loop_ids]).long(),
                col=torch.cat([neighboor_dst_unique_ids, self_loop_ids]).long(),
                sparse_sizes=(target_gid_1.size()[0], target_neighbor_count),
            )
        else:
            edge_index = SparseTensor(
                row=neighboor_src_unique_ids.long(),
                col=neighboor_dst_unique_ids.long(),
                sparse_sizes=(target_gid_1.size()[0], target_neighbor_count),
            )
        return edge_index
    elif framework_name == "dgl":
        if add_self_loop:
            csr_row_ptr, csr_col_ind = add_csr_self_loop(csr_row_ptr, csr_col_ind)
        block = dgl.create_block(
            (
                'csc',
                (
                    csr_row_ptr,
                    csr_col_ind,
                    torch.empty(0, dtype=torch.int),
                ),
            ),
            num_src_nodes=target_gid.size(0),
            num_dst_nodes=target_gid_1.size(0),
        )
        return block
    elif framework_name == "cugraph":
        if add_self_loop:
            csr_row_ptr, csr_col_ind = add_csr_self_loop(csr_row_ptr, csr_col_ind)
            max_num_neighbors = max_num_neighbors + 1
        return [csr_row_ptr, csr_col_ind, max_num_neighbors]
    else:
        assert framework_name == "wg"
        return [csr_row_ptr, csr_col_ind]
    return None


def layer_forward(layer, x_feat, x_target_feat, sub_graph):
    global framework_name
    if framework_name == "pyg":
        x_feat = layer((x_feat, x_target_feat), sub_graph)
    elif framework_name == "dgl":
        x_feat = layer(sub_graph, (x_feat, x_target_feat))
    elif framework_name == "cugraph":
        x_feat = layer(x_feat, sub_graph[0], sub_graph[1], sub_graph[2])
    elif framework_name == "wg":
        x_feat = layer(sub_graph[0], sub_graph[1], x_feat, x_target_feat)
    return x_feat


class HomoGNNModel(torch.nn.Module):
    def __init__(
        self,
        graph_structure: GraphStructure,
        node_embedding: WholeMemoryEmbedding,
        args,
    ):
        super().__init__()
        hidden_feat_dim = args.hiddensize
        self.graph_structure = graph_structure
        self.node_embedding = node_embedding
        self.num_layer = args.layernum
        self.hidden_feat_dim = args.hiddensize
        num_head = args.heads if (args.model == "gat") else 1
        assert hidden_feat_dim % num_head == 0
        in_feat_dim = self.node_embedding.shape[1]
        self.gnn_layers = create_gnn_layers(
            in_feat_dim,
            hidden_feat_dim,
            args.classnum,
            args.layernum,
            num_head,
            args.model,
        )
        self.mean_output = True if args.model == "gat" else False
        self.add_self_loop = True if args.model == "gat" else False
        self.gather_fn = WholeMemoryEmbeddingModule(self.node_embedding)
        self.dropout = args.dropout
        self.max_neighbors = parse_max_neighbors(args.layernum, args.neighbors)
        self.max_inference_neighbors = parse_max_neighbors(args.layernum, args.inferencesample)

    def forward(self, ids):
        global framework_name
        max_neighbors = self.max_neighbors if self.training else self.max_inference_neighbors
        ids = ids.to(self.graph_structure.csr_col_ind.dtype).cuda()
        (
            target_gids,
            edge_indice,
            csr_row_ptrs,
            csr_col_inds,
        ) = self.graph_structure.multilayer_sample_without_replacement(
            ids, max_neighbors
        )
        x_feat = self.gather_fn(target_gids[0], force_dtype=torch.float32)
        for i in range(self.num_layer):
            x_target_feat = x_feat[: target_gids[i + 1].numel()]
            sub_graph = create_sub_graph(
                target_gids[i],
                target_gids[i + 1],
                edge_indice[i],
                csr_row_ptrs[i],
                csr_col_inds[i],
                max_neighbors[self.num_layer - 1 - i],
                self.add_self_loop,
            )
            x_feat = layer_forward(
                self.gnn_layers[i],
                x_feat,
                x_target_feat,
                sub_graph,
            )
            if i != self.num_layer - 1:
                if framework_name == "dgl":
                    x_feat = x_feat.flatten(1)
                x_feat = F.relu(x_feat)
                x_feat = F.dropout(x_feat, self.dropout, training=self.training)
        if framework_name == "dgl" and self.mean_output:
            out_feat = x_feat.mean(1)
        else:
            out_feat = x_feat
        return out_feat
