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

import apex
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from apex.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
from ogb.linkproppred import Evaluator
from wg_torch import comm as comm
from wg_torch import embedding_ops as embedding_ops
from wg_torch import graph_ops as graph_ops
from wg_torch.wm_tensor import *

from wholegraph.torch import wholegraph_pytorch as wg

parser = OptionParser()
parser.add_option(
    "-r",
    "--root_dir",
    dest="root_dir",
    default="dataset",
    help="dataset root directory.",
)
parser.add_option(
    "-g", "--graph_name", dest="graph_name", default="ogbl-citation2", help="graph name"
)
parser.add_option(
    "-e", "--epochs", type="int", dest="epochs", default=1, help="number of epochs"
)
parser.add_option(
    "-b", "--batchsize", type="int", dest="batchsize", default=1024, help="batch size"
)
parser.add_option(
    "-n",
    "--neighbors",
    dest="neighbors",
    default="15,10,5",
    help="train neighboor sample count",
)
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
    default="wg",
    help="framework type, valid values are: dgl, pyg, wg",
)
parser.add_option("--heads", type="int", dest="heads", default=1, help="num heads")
parser.add_option(
    "-w",
    "--dataloaderworkers",
    type="int",
    dest="dataloaderworkers",
    default=8,
    help="number of workers for dataloader",
)
parser.add_option(
    "-d", "--dropout", type="float", dest="dropout", default=0.5, help="dropout"
)
parser.add_option("--lr", type="float", dest="lr", default=0.001, help="learning rate")
parser.add_option(
    "--use_nccl",
    action="store_true",
    dest="use_nccl",
    default=False,
    help="whether use nccl for embeddings, default False",
)

(options, args) = parser.parse_args()

use_chunked = True
use_host_memory = False


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


if options.framework == "dgl":
    import dgl
    from dgl.nn.pytorch.conv import SAGEConv, GATConv
elif options.framework == "pyg":
    from torch_sparse import SparseTensor
    from torch_geometric.nn import SAGEConv, GATConv
elif options.framework == "wg":
    from wg_torch.gnn.SAGEConv import SAGEConv
    from wg_torch.gnn.GATConv import GATConv


def create_gnn_layers(in_feat_dim, hidden_feat_dim, num_layer, num_head):
    gnn_layers = torch.nn.ModuleList()
    for i in range(num_layer):
        layer_output_dim = hidden_feat_dim // num_head
        layer_input_dim = in_feat_dim if i == 0 else hidden_feat_dim
        mean_output = True if i == num_layer - 1 else False
        if options.framework == "pyg":
            if options.model == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            elif options.model == "gat":
                concat = not mean_output
                gnn_layers.append(
                    GATConv(
                        layer_input_dim, layer_output_dim, heads=num_head, concat=concat
                    )
                )
            else:
                assert options.model == "gcn"
                gnn_layers.append(
                    SAGEConv(layer_input_dim, layer_output_dim, root_weight=False)
                )
        elif options.framework == "dgl":
            if options.model == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, "mean"))
            elif options.model == "gat":
                gnn_layers.append(
                    GATConv(
                        layer_input_dim,
                        layer_output_dim,
                        num_heads=num_head,
                        allow_zero_in_degree=True,
                    )
                )
            else:
                assert options.model == "gcn"
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, "gcn"))
        elif options.framework == "wg":
            if options.model == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            elif options.model == "gat":
                gnn_layers.append(
                    GATConv(
                        layer_input_dim,
                        layer_output_dim,
                        num_heads=num_head,
                        mean_output=mean_output,
                    )
                )
            else:
                assert options.model == "gcn"
                gnn_layers.append(
                    SAGEConv(layer_input_dim, layer_output_dim, aggregator="gcn")
                )
    return gnn_layers


def create_sub_graph(
    target_gid,
    target_gid_1,
    edge_data,
    csr_row_ptr,
    csr_col_ind,
    sample_dup_count,
    add_self_loop: bool,
):
    if options.framework == "pyg":
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
    elif options.framework == "dgl":
        if add_self_loop:
            self_loop_ids = torch.arange(
                0,
                target_gid_1.numel(),
                dtype=edge_data[0].dtype,
                device=target_gid.device,
            )
            block = dgl.create_block(
                (
                    torch.cat([edge_data[0], self_loop_ids]),
                    torch.cat([edge_data[1], self_loop_ids]),
                ),
                num_src_nodes=target_gid.size(0),
                num_dst_nodes=target_gid_1.size(0),
            )
        else:
            block = dgl.create_block(
                (edge_data[0], edge_data[1]),
                num_src_nodes=target_gid.size(0),
                num_dst_nodes=target_gid_1.size(0),
            )
        return block
    else:
        assert options.framework == "wg"
        return [csr_row_ptr, csr_col_ind, sample_dup_count]
    return None


def layer_forward(layer, x_feat, x_target_feat, sub_graph):
    if options.framework == "pyg":
        x_feat = layer((x_feat, x_target_feat), sub_graph)
    elif options.framework == "dgl":
        x_feat = layer(sub_graph, (x_feat, x_target_feat))
    elif options.framework == "wg":
        x_feat = layer(sub_graph[0], sub_graph[1], sub_graph[2], x_feat, x_target_feat)
    return x_feat


class EdgePredictionGNNModel(torch.nn.Module):
    def __init__(
        self, graph: graph_ops.HomoGraph, num_layer, hidden_feat_dim, max_neighbors: str
    ):
        super().__init__()
        self.graph = graph
        self.num_layer = num_layer
        self.hidden_feat_dim = hidden_feat_dim
        self.max_neighbors = parse_max_neighbors(num_layer, max_neighbors)
        num_head = options.heads if (options.model == "gat") else 1
        assert hidden_feat_dim % num_head == 0
        in_feat_dim = self.graph.node_feat_shape()[1]
        self.gnn_layers = create_gnn_layers(
            in_feat_dim, hidden_feat_dim, num_layer, num_head
        )
        self.mean_output = True if options.model == "gat" else False
        self.add_self_loop = True if options.model == "gat" else False
        self.gather_fn = embedding_ops.EmbeddingLookUpModule(need_backward=True)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_feat_dim, hidden_feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_feat_dim, hidden_feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_feat_dim, 1),
        )

    def gnn_forward(self, ids, exclude_edge_hashset=None):
        ids = ids.to(self.graph.id_type()).cuda()
        (
            target_gids,
            edge_indice,
            csr_row_ptrs,
            csr_col_inds,
            sample_dup_counts,
        ) = self.graph.unweighted_sample_without_replacement(
            ids, self.max_neighbors, exclude_edge_hashset=exclude_edge_hashset
        )
        x_feat = self.gather_fn(target_gids[0], self.graph.node_feat)
        # x_feat = self.graph.gather(target_gids[0])
        for i in range(self.num_layer):
            x_target_feat = x_feat[: target_gids[i + 1].numel()]
            sub_graph = create_sub_graph(
                target_gids[i],
                target_gids[i + 1],
                edge_indice[i],
                csr_row_ptrs[i],
                csr_col_inds[i],
                sample_dup_counts[i],
                self.add_self_loop,
            )
            x_feat = layer_forward(self.gnn_layers[i], x_feat, x_target_feat, sub_graph)
            if i != self.num_layer - 1:
                if options.framework == "dgl":
                    x_feat = x_feat.flatten(1)
                x_feat = F.relu(x_feat)
                # x_feat = F.dropout(x_feat, options.dropout, training=self.training)
        if options.framework == "dgl" and self.mean_output:
            out_feat = x_feat.mean(1)
        else:
            out_feat = x_feat
        return out_feat

    def predict(self, h_src, h_dst):
        return self.predictor(h_src * h_dst)

    def fullbatch_single_layer_forward(
        self, dist_homo_graph, i, input_feat, output_feat, batch_size
    ):
        start_node_id = (
            dist_homo_graph.node_count
            * wg.get_rank(dist_homo_graph.wm_comm)
            // wg.get_size(dist_homo_graph.wm_comm)
        )
        end_node_id = (
            dist_homo_graph.node_count
            * (wg.get_rank(dist_homo_graph.wm_comm) + 1)
            // wg.get_size(dist_homo_graph.wm_comm)
        )
        min_node_count = dist_homo_graph.node_count // wg.get_size(
            dist_homo_graph.wm_comm
        )
        total_node_count = end_node_id - start_node_id
        batch_count = max((min_node_count + batch_size - 1) // batch_size, 1)
        last_batchsize = total_node_count - (batch_count - 1) * batch_size
        embedding_lookup_fn = embedding_ops.EmbeddingLookupFn.apply
        for batch_id in range(batch_count):
            current_batchsize = (
                last_batchsize if batch_id == batch_count - 1 else batch_size
            )
            batch_start_node_id = start_node_id + batch_id * batch_size
            target_ids = torch.arange(
                batch_start_node_id,
                batch_start_node_id + current_batchsize,
                dtype=dist_homo_graph.edges_csr_col.dtype,
                device="cuda",
            )
            (
                neighboor_gids_offset,
                neighboor_gids_vdata,
                neighboor_src_lids,
            ) = graph_ops.unweighted_sample_without_replacement_single_layer(
                target_ids,
                dist_homo_graph.edges_csr_row,
                dist_homo_graph.edges_csr_col,
                -1,
            )
            (
                unique_gids,
                neighbor_raw_to_unique_mapping,
                unique_output_neighbor_count,
            ) = torch.ops.wholegraph.append_unique(target_ids, neighboor_gids_vdata)
            csr_row_ptr = neighboor_gids_offset
            csr_col_ind = neighbor_raw_to_unique_mapping
            sample_dup_count = unique_output_neighbor_count
            neighboor_count = neighboor_gids_vdata.size()[0]
            edge_indice_i = torch.cat(
                [
                    torch.reshape(neighbor_raw_to_unique_mapping, (1, neighboor_count)),
                    torch.reshape(neighboor_src_lids, (1, neighboor_count)),
                ]
            )
            target_ids_i = unique_gids
            x_feat = embedding_lookup_fn(target_ids_i, input_feat)
            sub_graph = create_sub_graph(
                target_ids_i,
                target_ids,
                edge_indice_i,
                csr_row_ptr,
                csr_col_ind,
                sample_dup_count,
                self.add_self_loop,
            )
            x_target_feat = x_feat[: target_ids.numel()]
            x_feat = layer_forward(self.gnn_layers[i], x_feat, x_target_feat, sub_graph)
            if i != self.num_layer - 1:
                if options.framework == "dgl":
                    x_feat = x_feat.flatten(1)
                x_feat = F.relu(x_feat)
            else:
                if options.framework == "dgl" and self.mean_output:
                    x_feat = x_feat.mean(1)
            embedding_ops.embedding_2d_sub_tensor_assign(
                x_feat, output_feat, batch_start_node_id
            )

    def forward(self, src_ids, pos_dst_ids, neg_dst_ids):
        assert src_ids.shape == pos_dst_ids.shape and src_ids.shape == neg_dst_ids.shape
        id_count = src_ids.size(0)
        ids = torch.cat([src_ids, pos_dst_ids, neg_dst_ids])
        # add both forward and reverse edge into hashset
        exclude_edge_hashset = torch.ops.wholegraph.create_edge_hashset(
            torch.cat([src_ids, pos_dst_ids]), torch.cat([pos_dst_ids, src_ids])
        )
        ids_unique, reverse_map = torch.unique(ids, return_inverse=True)
        out_feat_unique = self.gnn_forward(
            ids_unique, exclude_edge_hashset=exclude_edge_hashset
        )
        out_feat = torch.nn.functional.embedding(reverse_map, out_feat_unique)
        src_feat, pos_dst_feat, neg_dst_feat = torch.split(out_feat, id_count)
        scores = self.predict(
            torch.cat([src_feat, src_feat]), torch.cat([pos_dst_feat, neg_dst_feat])
        )
        return scores[:id_count], scores[id_count:]


def compute_mrr(model, node_emb, src, dst, neg_dst, batch_size=1024):
    rr = torch.zeros(src.shape[0])
    embedding_lookup_fn = embedding_ops.EmbeddingLookupFn.apply
    evaluator = Evaluator(name="ogbl-citation2")
    preds = []
    for start in range(0, src.shape[0], batch_size):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = embedding_lookup_fn(src[start:end], node_emb)[:, None, :]
        h_dst = embedding_lookup_fn(all_dst.view(-1), node_emb).view(*all_dst.shape, -1)
        pred = model.predict(h_src, h_dst).squeeze(-1)
        relevance = torch.zeros(*pred.shape, dtype=torch.bool)
        relevance[:, 0] = True
        rr[start:end] = MF.retrieval_reciprocal_rank(pred, relevance)
        preds += [pred]
    all_pred = torch.cat(preds)
    pos_pred = all_pred[:, :1].squeeze(1)
    neg_pred = all_pred[:, 1:]
    ogb_mrr = (
        evaluator.eval(
            {
                "y_pred_pos": pos_pred,
                "y_pred_neg": neg_pred,
            }
        )["mrr_list"]
        .mean()
        .item()
    )
    return rr.mean().item(), ogb_mrr


@torch.no_grad()
def evaluate(model: EdgePredictionGNNModel, dist_homo_graph, edge_split):
    global use_chunked
    global use_host_memory
    model.eval()
    embedding = dist_homo_graph.node_feat
    node_feats = [None, None]
    wm_tensor_type = get_intra_node_wm_tensor_type(use_chunked, use_host_memory)
    node_feats[0] = create_wm_tensor(
        dist_homo_graph.wm_comm,
        [embedding.shape[0], options.hiddensize],
        [],
        embedding.dtype,
        wm_tensor_type,
    )
    if options.layernum > 1:
        node_feats[1] = create_wm_tensor(
            dist_homo_graph.wm_comm,
            [embedding.shape[0], options.hiddensize],
            [],
            embedding.dtype,
            wm_tensor_type,
        )
    output_feat = node_feats[0]
    input_feat = embedding
    del embedding
    for i in range(options.layernum):
        model.fullbatch_single_layer_forward(
            dist_homo_graph, i, input_feat, output_feat, 1024
        )
        wg.barrier(dist_homo_graph.wm_comm)
        input_feat = output_feat
        output_feat = node_feats[(i + 1) % 2]
    del output_feat
    del node_feats[1]
    del node_feats[0]
    del node_feats
    dgl_mrr_results = []
    ogb_mrr_results = []
    for split in ["valid", "test"]:
        src = torch.from_numpy(edge_split[split]["source_node"]).cuda()
        dst = torch.from_numpy(edge_split[split]["target_node"]).cuda()
        neg_dst = torch.from_numpy(edge_split[split]["target_node_neg"]).cuda()
        dgl_mrr, ogb_mrr = compute_mrr(model, input_feat, src, dst, neg_dst)
        dgl_mrr_results.append(dgl_mrr)
        ogb_mrr_results.append(ogb_mrr)
    return dgl_mrr_results, ogb_mrr_results


def train(dist_homo_graph, model, optimizer):
    print("start training...")
    train_step = 0
    epoch = 0
    train_start_time = time.time()
    while epoch < options.epochs:
        epoch_iter_count = dist_homo_graph.start_iter(options.batchsize)
        if comm.get_rank() == 0:
            print("%d steps for epoch %d." % (epoch_iter_count, epoch))
        iter_id = 0
        while iter_id < epoch_iter_count:
            src_nid, pos_dst_nid = dist_homo_graph.get_train_edge_batch(iter_id)
            # neg_dst_nid = torch.randint_like(src_nid, 0, dist_homo_graph.node_count)
            neg_dst_nid = dist_homo_graph.per_source_negative_sample(src_nid)
            optimizer.zero_grad()
            model.train()
            pos_score, neg_score = model(src_nid, pos_dst_nid, neg_dst_nid)
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            score = torch.cat([pos_score, neg_score])
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            loss.backward()
            optimizer.step()
            embedding_ops.run_optimizers(options.lr * 0.1)
            if comm.get_rank() == 0 and train_step % 100 == 0:
                print(
                    "[%s] [LOSS] step=%d, loss=%f"
                    % (
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        train_step,
                        loss.cpu().item(),
                    )
                )
            train_step = train_step + 1
            iter_id = iter_id + 1
        epoch = epoch + 1
    comm.synchronize()
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    if comm.get_rank() == 0:
        print(
            "[%s] [TRAIN_TIME] train time is %.2f seconds"
            % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), train_time)
        )
        print("[EPOCH_TIME] %.2f seconds" % (train_time / options.epochs,))


def main():
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
    print("Rank=%d, local_rank=%d" % (local_rank, comma.Get_rank()))
    dev_count = torch.cuda.device_count()
    assert dev_count > 0
    assert local_size <= dev_count
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    wm_comm = create_intra_node_communicator(
        comma.Get_rank(), comma.Get_size(), local_size
    )
    if comma.Get_rank() == 0:
        print("Framework=%s, Model=%s" % (options.framework, options.model))
    embedding_ops.init_embedding_backward_env(
        wm_comm, embedding_ops.EmbeddingLazyAdamOptimizer()
    )

    edge_split = graph_ops.load_pickle_link_pred_data(
        options.root_dir, options.graph_name, True
    )

    dist_homo_graph = graph_ops.HomoGraph()
    global use_chunked
    global use_host_memory
    dist_homo_graph.load(
        options.root_dir,
        options.graph_name,
        wm_comm,
        use_chunked,
        use_host_memory,
        feat_dtype=None,
        id_dtype=None,
        ignore_embeddings=["paper"],
        link_pred_task=True,
    )
    print("Rank=%d, Graph loaded." % (comma.Get_rank(),))
    dist_homo_graph.create_node_embedding(
        "paper",
        use_chunked=use_chunked,
        use_host_memory=use_host_memory,
        use_nccl=options.use_nccl,
    )
    lt = get_local_tensor(dist_homo_graph.node_feat)
    torch.nn.init.xavier_uniform_(lt)
    del lt
    dist_homo_graph.node_feat = embedding_ops.TrainableEmbedding(
        dist_homo_graph.node_feat
    )

    # dist_homo_graph.node_feat = embedding_ops.TrainableEmbedding(
    #    dist_homo_graph.node_feat, chunked=use_chunked, use_host_memory=use_host_memory
    # )
    raw_model = EdgePredictionGNNModel(
        dist_homo_graph, options.layernum, options.hiddensize, options.neighbors
    )
    print("Rank=%d, model created." % (comma.Get_rank(),))
    raw_model.cuda()
    print("Rank=%d, model movded to cuda." % (comma.Get_rank(),))
    model = DDP(raw_model, delay_allreduce=True)
    optimizer = apex.optimizers.FusedAdam(
        model.parameters(), lr=options.lr, weight_decay=5e-4
    )
    print("Rank=%d, optimizer created." % (comma.Get_rank(),))

    # graph_ops.load_homo_graph_model_state('checkpoints', 'model', model, dist_homo_graph, -1)

    dgl_mrr, ogb_mrr = evaluate(raw_model, dist_homo_graph, edge_split)
    if comm.get_rank() == 0:
        print(
            "Validation DGL MRR:",
            dgl_mrr[0],
            "Test DGL MRR:",
            dgl_mrr[1],
            "Validation OGB MRR:",
            ogb_mrr[0],
            "Test OGB MRR:",
            ogb_mrr[1],
        )

    train(dist_homo_graph, model, optimizer)

    # graph_ops.save_homo_graph_model_state('checkpoints', 'model', model, dist_homo_graph, options.epochs)

    dgl_mrr, ogb_mrr = evaluate(raw_model, dist_homo_graph, edge_split)
    if comm.get_rank() == 0:
        print(
            "Validation DGL MRR:",
            dgl_mrr[0],
            "Test DGL MRR:",
            dgl_mrr[1],
            "Validation OGB MRR:",
            ogb_mrr[0],
            "Test OGB MRR:",
            ogb_mrr[1],
        )

    embedding_ops.finalize_embedding_backward_env()
    wg.finalize_lib()
    print("Rank=%d, wholegraph shutdown." % (comma.Get_rank(),))


if __name__ == "__main__":
    main()
