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

# graph related operations

import os
import re
from enum import IntEnum
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset
from wg_torch import comm
from wg_torch import embedding_ops as embedding_ops
from wg_torch.wm_tensor import *

from wholegraph.torch import wholegraph_pytorch as wg


def load_meta_file(save_dir, graph_name):
    meta_file_name = graph_name + "_meta.json"
    meta_file_path = os.path.join(save_dir, meta_file_name)
    import json

    meta_data = json.load(open(meta_file_path, "r"))
    return meta_data


def save_meta_file(save_dir, meta_data, graph_name):
    meta_file_name = graph_name + "_meta.json"
    meta_file_path = os.path.join(save_dir, meta_file_name)
    import json

    json.dump(meta_data, open(meta_file_path, "w"))


numpy_dtype_to_string_dict = {
    np.dtype("float16"): "half",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "double",
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
}
string_to_pytorch_dtype_dict = {
    "half": torch.float16,
    "float32": torch.float32,
    "double": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}
pytorch_dtype_string_to_dict = {
    "torch.float16": "half",
    "torch.float32": "float32",
    "torch.float64": "double",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
}


def numpy_dtype_to_string(dtype: torch.dtype):
    if dtype not in numpy_dtype_to_string_dict.keys():
        print("dtype type %s %s not in dict" % (type(dtype), dtype))
        raise ValueError
    return numpy_dtype_to_string_dict[dtype]


def pytorch_dtype_to_string(dtype: torch.dtype):
    if str(dtype) not in pytorch_dtype_string_to_dict.keys():
        print("dtype type %s %s not in dict" % (type(dtype), dtype))
        raise ValueError
    return pytorch_dtype_string_to_dict[str(dtype)]


def string_to_pytorch_dtype(dtype_str: str):
    if dtype_str not in string_to_pytorch_dtype_dict.keys():
        print("string dtype %s not in dict" % (dtype_str,))
        raise ValueError
    return string_to_pytorch_dtype_dict[dtype_str]


def parse_part_file(part_file_name: str, prefix: str):
    if not part_file_name.startswith(prefix):
        return None, None
    if part_file_name == prefix:
        return 0, 1
    pattern = re.compile("_part_(\d+)_of_(\d+)")
    matches = pattern.match(part_file_name[len(prefix) :])
    int_tuple = matches.groups()
    if len(int_tuple) != 2:
        return None, None
    return int(int_tuple[0]), int(int_tuple[1])


def get_part_filename(prefix: str, idx: int = 0, count: int = 1):
    filename = prefix
    filename += "_part_%d_of_%d" % (idx, count)
    return filename


def check_part_files_in_path(save_dir, prefix):
    valid_files = 0
    total_file_count = 0
    for filename in os.listdir(save_dir):
        if not os.path.isfile(os.path.join(save_dir, filename)):
            continue
        if not filename.startswith(prefix):
            continue
        idx, count = parse_part_file(filename, prefix)
        if idx is None or count is None:
            continue
        valid_files += 1
        if total_file_count == 0:
            total_file_count = count
        else:
            raise FileExistsError(
                "prefix %s both count=%d and count=%d exist."
                % (prefix, total_file_count, count)
            )
    if valid_files == total_file_count:
        return total_file_count
    if total_file_count != valid_files:
        raise FileNotFoundError(
            "prefix %s count=%d but got only %d files."
            % (prefix, total_file_count, valid_files)
        )
    return None


def check_data_integrity(save_dir, graph_name):
    meta_file_name = graph_name + "_meta.json"
    meta_file_path = os.path.join(save_dir, meta_file_name)
    if not os.path.exists(meta_file_path):
        return False
    if not os.path.isfile(meta_file_path):
        return False
    meta_data = load_meta_file(save_dir, graph_name)
    if meta_data is None:
        return False

    for node_type in meta_data["nodes"]:
        if node_type["has_emb"]:
            node_emb_prefix = node_type["emb_file_prefix"]
            emb_file_count = check_part_files_in_path(save_dir, node_emb_prefix)
            if emb_file_count == 0:
                return False

    for edge_type in meta_data["edges"]:
        edge_list_prefix = edge_type["edge_list_prefix"]
        edge_file_count = check_part_files_in_path(save_dir, edge_list_prefix)
        if edge_file_count == 0:
            return False
        if edge_type["has_emb"]:
            edge_emb_prefix = edge_type["emb_file_prefix"]
            emb_file_count = check_part_files_in_path(save_dir, edge_emb_prefix)
            if emb_file_count == 0:
                return False

    return True


def graph_name_normalize(graph_name: str):
    return graph_name.replace("-", "_")


def download_and_convert_papers100m(save_dir, ogb_root_dir="dataset"):
    graph_name = "papers100m"
    from ogb.nodeproppred import NodePropPredDataset

    if check_data_integrity(save_dir, graph_name):
        return
    dataset = NodePropPredDataset(name="ogbn-papers100M", root=ogb_root_dir)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
    graph, label = dataset[0]
    for name in ["num_nodes", "edge_index", "node_feat", "edge_feat"]:
        if name not in graph.keys():
            raise ValueError(
                "graph has no key %s, graph.keys()= %s" % (name, graph.keys())
            )
    num_nodes = graph["num_nodes"]
    edge_index = graph["edge_index"]
    node_feat = graph["node_feat"]
    edge_feat = graph["edge_feat"]
    if isinstance(num_nodes, np.int64) or isinstance(num_nodes, np.int32):
        num_nodes = num_nodes.item()
    if (
        not isinstance(edge_index, np.ndarray)
        or len(edge_index.shape) != 2
        or edge_index.shape[0] != 2
    ):
        raise TypeError("edge_index is not numpy.ndarray of shape (2, x)")
    num_edges = edge_index.shape[1]
    assert node_feat is not None
    if (
        not isinstance(node_feat, np.ndarray)
        or len(node_feat.shape) != 2
        or node_feat.shape[0] != num_nodes
    ):
        raise ValueError("node_feat is not numpy.ndarray of shape (num_nodes, x)")
    node_feat_dim = node_feat.shape[1]
    node_feat_name_prefix = "papers100m_node_feat_paper"
    edge_index_name_prefix = "papers100m_edge_index_paper_cites_paper"

    nodes = [
        {
            "name": "paper",
            "has_emb": True,
            "emb_file_prefix": node_feat_name_prefix,
            "num_nodes": num_nodes,
            "emb_dim": node_feat_dim,
            "dtype": numpy_dtype_to_string(node_feat.dtype),
        }
    ]
    edges = [
        {
            "src": "paper",
            "dst": "paper",
            "rel": "cites",
            "has_emb": False,
            "edge_list_prefix": edge_index_name_prefix,
            "num_edges": num_edges,
            "dtype": numpy_dtype_to_string(np.dtype("int32")),
            "directed": True,
        }
    ]
    meta_json = {"nodes": nodes, "edges": edges}
    save_meta_file(save_dir, meta_json, graph_name)
    train_label = label[train_idx]
    valid_label = label[valid_idx]
    test_label = label[test_idx]
    data_and_label = {
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "train_label": train_label,
        "valid_label": valid_label,
        "test_label": test_label,
    }
    import pickle

    with open(os.path.join(save_dir, graph_name + "_data_and_label.pkl"), "wb") as f:
        pickle.dump(data_and_label, f)
    print("saving node feature...")
    with open(
        os.path.join(save_dir, get_part_filename(node_feat_name_prefix)), "wb"
    ) as f:
        node_feat.tofile(f)
    print("converting edge index...")
    edge_index_int32 = np.transpose(edge_index).astype(np.int32)
    print("saving edge index...")
    with open(
        os.path.join(save_dir, get_part_filename(edge_index_name_prefix)), "wb"
    ) as f:
        edge_index_int32.tofile(f)

    assert edge_feat is None


def download_and_convert_citation2(save_dir, ogb_root_dir="dataset"):
    graph_name = "citation2"
    if check_data_integrity(save_dir, graph_name):
        return
    from ogb.linkproppred import LinkPropPredDataset

    dataset = LinkPropPredDataset(name="ogbl-citation2", root=ogb_root_dir)
    graph = dataset[0]
    for name in ["num_nodes", "edge_index", "node_feat", "edge_feat", "node_year"]:
        if name not in graph.keys():
            raise ValueError(
                "graph has no key %s, graph.keys()= %s" % (name, graph.keys())
            )
    num_nodes = graph["num_nodes"]
    edge_index = graph["edge_index"]
    node_feat = graph["node_feat"]
    edge_feat = graph["edge_feat"]
    node_year = graph["node_year"]
    split_edge = dataset.get_edge_split()
    if isinstance(num_nodes, np.int64) or isinstance(num_nodes, np.int32):
        num_nodes = num_nodes.item()
    if (
        not isinstance(edge_index, np.ndarray)
        or len(edge_index.shape) != 2
        or edge_index.shape[0] != 2
    ):
        raise TypeError("edge_index is not numpy.ndarray of shape (2, x)")
    num_edges = edge_index.shape[1]
    assert node_feat is not None
    if (
        not isinstance(node_feat, np.ndarray)
        or len(node_feat.shape) != 2
        or node_feat.shape[0] != num_nodes
    ):
        raise ValueError("node_feat is not numpy.ndarray of shape (num_nodes, x)")
    node_feat_dim = node_feat.shape[1]
    node_feat_name_prefix = "citation2_node_feat_paper"
    edge_index_name_prefix = "citation2_edge_index_paper_cites_paper"

    nodes = [
        {
            "name": "paper",
            "has_emb": True,
            "emb_file_prefix": node_feat_name_prefix,
            "num_nodes": num_nodes,
            "emb_dim": node_feat_dim,
            "dtype": numpy_dtype_to_string(node_feat.dtype),
        }
    ]
    edges = [
        {
            "src": "paper",
            "dst": "paper",
            "rel": "cites",
            "has_emb": False,
            "edge_list_prefix": edge_index_name_prefix,
            "num_edges": num_edges,
            "dtype": numpy_dtype_to_string(np.dtype("int32")),
            "directed": True,
        }
    ]
    meta_json = {"nodes": nodes, "edges": edges}
    save_meta_file(save_dir, meta_json, graph_name)

    valid_test_edges = {"valid": split_edge["valid"], "test": split_edge["test"]}

    train_edges = np.stack(
        (split_edge["train"]["source_node"], split_edge["train"]["target_node"])
    )
    assert (train_edges == edge_index).all()

    import pickle

    with open(os.path.join(save_dir, graph_name + "_node_year.pkl"), "wb") as f:
        pickle.dump(node_year, f)
    with open(
        os.path.join(save_dir, graph_name + "_link_prediction_test_valid.pkl"), "wb"
    ) as f:
        pickle.dump(valid_test_edges, f)
    print("saving node feature...")
    with open(
        os.path.join(save_dir, get_part_filename(node_feat_name_prefix)), "wb"
    ) as f:
        node_feat.tofile(f)
    print("converting edge index...")
    edge_index_int32 = np.transpose(edge_index).astype(np.int32)
    print("saving edge index...")
    with open(
        os.path.join(save_dir, get_part_filename(edge_index_name_prefix)), "wb"
    ) as f:
        edge_index_int32.tofile(f)

    assert edge_feat is None


class GraphExtractType(IntEnum):
    EQUAL = 1
    LESS = 2
    LESS_OR_EQUAL = 3
    GREAT = 4
    GREAT_OR_EQUAL = 5


def extract_subgraph_with_filter(
    extract_type: GraphExtractType,
    target_gid: torch.Tensor,
    filter_target_value: torch.Tensor,
    edges_csr_row: Union[torch.Tensor, wg.ChunkedTensor],
    edges_csr_col: Union[torch.Tensor, wg.ChunkedTensor],
    edges_value: Union[torch.Tensor, wg.ChunkedTensor],
    need_value: bool = False,
):
    extract_type_int = int(extract_type)
    need_value_int = int(need_value)
    if isinstance(edges_csr_row, torch.Tensor):
        return torch.ops.wholegraph.extract_subgraph_with_filter(
            target_gid,
            filter_target_value,
            edges_csr_row,
            edges_csr_col,
            edges_value,
            extract_type_int,
            need_value_int,
        )
    else:
        return torch.ops.wholegraph.extract_subgraph_with_filter_chunked(
            target_gid,
            filter_target_value,
            edges_csr_row.get_ptr(),
            edges_csr_col.get_ptr(),
            edges_value.get_ptr(),
            extract_type_int,
            need_value_int,
        )


def unweighted_sample_without_replacement_single_layer(
    target_gid: torch.Tensor,
    edges_csr_row: Union[torch.Tensor, wg.ChunkedTensor],
    edges_csr_col: Union[torch.Tensor, wg.ChunkedTensor],
    max_neighbor: int,
):
    is_chunked = isinstance(edges_csr_row, wg.ChunkedTensor)
    if is_chunked:
        (
            neighboor_gids_offset,
            neighboor_gids_vdata,
            neighboor_src_lids,
        ) = torch.ops.wholegraph.unweighted_sample_without_replacement_chunked(
            target_gid, edges_csr_row.get_ptr(), edges_csr_col.get_ptr(), max_neighbor
        )
    else:
        (
            neighboor_gids_offset,
            neighboor_gids_vdata,
            neighboor_src_lids,
        ) = torch.ops.wholegraph.unweighted_sample_without_replacement(
            target_gid, edges_csr_row, edges_csr_col, max_neighbor
        )
    return neighboor_gids_offset, neighboor_gids_vdata, neighboor_src_lids


def weighted_sample_without_replacement_single_layer(
    target_gid: torch.Tensor,
    edges_csr_row: Union[torch.Tensor, wg.ChunkedTensor],
    edges_csr_col: Union[torch.Tensor, wg.ChunkedTensor],
    edges_csr_weight: Union[torch.Tensor, wg.ChunkedTensor],
    max_neighbor: int,
    edges_csr_local_sorted_map_indices: Union[torch.Tensor, wg.ChunkedTensor] = None,
):
    is_chunked = isinstance(edges_csr_row, wg.ChunkedTensor)
    if is_chunked:
        edges_csr_local_sorted_map_indices_ptr = (
            None
            if edges_csr_local_sorted_map_indices == None
            else edges_csr_local_sorted_map_indices.get_ptr()
        )
        (
            neighboor_gids_offset,
            neighboor_gids_vdata,
            neighboor_src_lids,
        ) = torch.ops.wholegraph.weighted_sample_without_replacement_chunked(
            target_gid,
            edges_csr_row.get_ptr(),
            edges_csr_col.get_ptr(),
            edges_csr_weight.get_ptr(),
            max_neighbor,
            edges_csr_local_sorted_map_indices_ptr,
        )
    else:
        (
            neighboor_gids_offset,
            neighboor_gids_vdata,
            neighboor_src_lids,
        ) = torch.ops.wholegraph.weighted_sample_without_replacement(
            target_gid,
            edges_csr_row,
            edges_csr_col,
            edges_csr_weight,
            max_neighbor,
            edges_csr_local_sorted_map_indices,
        )
    return neighboor_gids_offset, neighboor_gids_vdata, neighboor_src_lids


def filter_edges(
    src_gids: torch.Tensor,
    neighboor_gids_offset: torch.Tensor,
    neighboor_gids_vdata: torch.Tensor,
    exclude_edge_hashset: torch.Tensor,
):
    # neighboor_gids_offset, neighboor_gids_vdata, neighboor_src_lids
    return torch.ops.wholegraph.filter_csr_edges(
        src_gids, neighboor_gids_offset, neighboor_gids_vdata, exclude_edge_hashset
    )


class HomoGraph(object):
    def __init__(self):
        self.node_feat = None
        self.edges_csr_row = None
        self.edges_csr_col = None
        self.edge_feat = None
        self.node_count = None
        self.edge_count = None
        self.meta_data = None
        self.is_chunked = True
        self.use_host_memory = False
        self.node_info = None
        self.edge_info = None
        self.wm_comm = None
        self.wm_nccl_embedding_comm = None

    def id_type(self):
        return self.id_dtype

    def node_feat_dtype(self):
        return self.feat_dtype

    def node_feat_shape(self):
        if isinstance(self.node_feat, embedding_ops.TrainableEmbedding):
            return self.node_feat.embedding.shape
        else:
            return self.node_feat.shape

    def load(
        self,
        dataset_dir: str,
        graph_name: str,
        wm_comm: int,
        use_chunked: bool,
        use_host_memory: bool = False,
        wm_nccl_embedding_comm: Union[int, None] = None,
        feat_dtype: Union[torch.dtype, None] = None,
        id_dtype: Union[torch.dtype, None] = None,
        ignore_embeddings: Union[list, None] = None,
        link_pred_task: bool = False,
    ):
        self.wm_comm = wm_comm
        self.wm_nccl_embedding_comm = wm_nccl_embedding_comm
        self.is_chunked = use_chunked
        self.use_host_memory = use_host_memory
        normalized_graph_name = graph_name_normalize(graph_name)
        save_dir = os.path.join(dataset_dir, normalized_graph_name, "converted")
        if not check_data_integrity(save_dir, normalized_graph_name):
            print(
                "path %s doesn't contain all the data for %s" % (save_dir, graph_name)
            )
            raise FileNotFoundError
        self.meta_data = load_meta_file(save_dir, normalized_graph_name)
        nodes = self.meta_data["nodes"]
        edges = self.meta_data["edges"]
        assert len(nodes) == 1
        assert len(edges) == 1
        self.node_info = nodes[0]
        self.edge_info = edges[0]
        self.node_count = nodes[0]["num_nodes"]
        data_edge_count = edges[0]["num_edges"]

        if id_dtype is None:
            id_dtype = string_to_pytorch_dtype(edges[0]["dtype"])
        self.id_dtype = id_dtype

        wm_tensor_type = get_intra_node_wm_tensor_type(use_chunked, use_host_memory)
        self.edges_csr_row = create_wm_tensor_from_file(
            [self.node_count + 1],
            torch.int64,
            self.wm_comm,
            os.path.join(save_dir, "homograph_csr_row_ptr"),
            wm_tensor_type,
        )
        self.edges_csr_col = create_wm_tensor_from_file(
            [],
            torch.int32,
            self.wm_comm,
            os.path.join(save_dir, "homograph_csr_col_idx"),
            wm_tensor_type,
        )
        self.edge_count = self.edges_csr_col.shape[0]

        if nodes[0]["has_emb"] and (
            ignore_embeddings is None or nodes[0]["name"] not in ignore_embeddings
        ):
            embedding_dim = nodes[0]["emb_dim"]
            src_dtype = string_to_pytorch_dtype(nodes[0]["dtype"])
            if feat_dtype is None:
                feat_dtype = src_dtype
            else:
                assert feat_dtype == src_dtype
            self.feat_dtype = feat_dtype
            node_emb_file_prefix = os.path.join(save_dir, nodes[0]["emb_file_prefix"])

            if self.wm_nccl_embedding_comm is None:
                self.node_feat = create_wm_tensor_from_file(
                    [self.node_count, embedding_dim],
                    feat_dtype,
                    self.wm_comm,
                    node_emb_file_prefix,
                    wm_tensor_type,
                    1,
                )
            else:
                self.node_feat = create_wm_tensor_from_file(
                    [self.node_count, embedding_dim],
                    feat_dtype,
                    self.wm_nccl_embedding_comm,
                    node_emb_file_prefix,
                    WmTensorType.NCCL,
                    1,
                )

        if link_pred_task is True:
            self.prepare_train_edges()
            self.create_edges_jump_coo_row()

    def create_node_embedding(
        self,
        node_name,
        use_chunked: bool = True,
        use_host_memory: bool = False,
        embedding_dim=None,
        use_nccl: bool = False,
    ):
        nodes = self.meta_data["nodes"]
        assert self.node_feat is None
        assert nodes[0]["name"] == node_name
        self.feat_dtype = torch.float32
        if embedding_dim is None:
            assert nodes[0]["has_emb"] is True
            embedding_dim = nodes[0]["emb_dim"]
        if use_nccl:
            wm_tensor_type = WmTensorType.NCCL
        else:
            wm_tensor_type = get_intra_node_wm_tensor_type(use_chunked, use_host_memory)
        self.node_feat = create_wm_tensor(
            self.wm_comm
            if self.wm_nccl_embedding_comm is None
            else self.wm_nccl_embedding_comm,
            [self.node_count, embedding_dim],
            [],
            self.feat_dtype,
            wm_tensor_type,
        )
        if use_host_memory and not use_nccl:
            self.node_feat = wg.get_tensor_view(
                self.node_feat, torch.device("cuda", torch.cuda.current_device())
            )

    def unweighted_sample_without_replacement(
        self, node_ids, max_neighbors, exclude_edge_hashset=None
    ):
        hops = len(max_neighbors)
        sample_dup_count = [None] * hops
        edge_indice = [None] * hops
        csr_row_ptr = [None] * hops
        csr_col_ind = [None] * hops
        target_gids = [None] * (hops + 1)
        target_gids[hops] = node_ids
        for i in range(hops - 1, -1, -1):
            (
                neighboor_gids_offset,
                neighboor_gids_vdata,
                neighboor_src_lids,
            ) = unweighted_sample_without_replacement_single_layer(
                target_gids[i + 1],
                self.edges_csr_row,
                self.edges_csr_col,
                max_neighbors[hops - i - 1],
            )
            if exclude_edge_hashset is not None:
                (
                    neighboor_gids_offset,
                    neighboor_gids_vdata,
                    neighboor_src_lids,
                ) = filter_edges(
                    target_gids[i + 1],
                    neighboor_gids_offset,
                    neighboor_gids_vdata,
                    exclude_edge_hashset,
                )
            (
                unique_gids,
                neighbor_raw_to_unique_mapping,
                unique_output_neighbor_count,
            ) = torch.ops.wholegraph.append_unique(
                target_gids[i + 1], neighboor_gids_vdata
            )
            #### no unique
            # unique_gids = torch.cat([target_gids[i + 1], neighboor_gids_vdata])
            # neighbor_raw_to_unique_mapping = torch.arange(target_gids[i + 1].shape[0], neighboor_gids_vdata.shape[0] + target_gids[i + 1].shape[0], dtype=torch.int32, device='cuda')
            # unique_output_neighbor_count = torch.zeros(size=(neighboor_gids_vdata.shape[0] + target_gids[i + 1].shape[0], ), dtype=torch.int32, device='cuda')
            #### No unique between neighbor and targets
            # unique_gids, neighbor_raw_to_unique_mapping, unique_output_neighbor_count = torch.ops.wholegraph.append_unique(
            #    target_gids[i + 1][0:0], neighboor_gids_vdata)
            # unique_gids = torch.cat([target_gids[i + 1], unique_gids])
            # neighbor_raw_to_unique_mapping = neighbor_raw_to_unique_mapping + target_gids[i + 1].shape[0]
            # unique_output_neighbor_count = torch.cat([torch.zeros(size=(target_gids[i + 1].shape[0], ), dtype=torch.int32, device='cuda'), unique_output_neighbor_count])
            ####
            csr_row_ptr[i] = neighboor_gids_offset
            csr_col_ind[i] = neighbor_raw_to_unique_mapping
            sample_dup_count[i] = unique_output_neighbor_count
            neighboor_count = neighboor_gids_vdata.size()[0]
            edge_indice[i] = torch.cat(
                [
                    torch.reshape(neighbor_raw_to_unique_mapping, (1, neighboor_count)),
                    torch.reshape(neighboor_src_lids, (1, neighboor_count)),
                ]
            )
            target_gids[i] = unique_gids
        return target_gids, edge_indice, csr_row_ptr, csr_col_ind, sample_dup_count

    def weighted_sample_without_replacement(
        self,
        node_ids,
        max_neighbors,
        csr_weight: Union[torch.Tensor, wg.ChunkedTensor],
        csr_local_sorted_map_indices: Union[torch.Tensor, wg.ChunkedTensor] = None,
        exclude_edge_hashset=None,
    ):
        if type(csr_weight) != type(self.edges_csr_col):
            raise TypeError(
                "  the type of csr_weight should be the same as that of self.edges_csr_col , but csr_weight's type  is {} while self.edges_csr_col is {} ".format(
                    type(csr_weight), type(self.edges_csr_col)
                )
            )
        hops = len(max_neighbors)
        sample_dup_count = [None] * hops
        edge_indice = [None] * hops
        csr_row_ptr = [None] * hops
        csr_col_ind = [None] * hops
        target_gids = [None] * (hops + 1)
        target_gids[hops] = node_ids
        for i in range(hops - 1, -1, -1):
            (
                neighboor_gids_offset,
                neighboor_gids_vdata,
                neighboor_src_lids,
            ) = weighted_sample_without_replacement_single_layer(
                target_gids[i + 1],
                self.edges_csr_row,
                self.edges_csr_col,
                csr_weight,
                max_neighbors[hops - i - 1],
                csr_local_sorted_map_indices,
            )
            if exclude_edge_hashset is not None:
                (
                    neighboor_gids_offset,
                    neighboor_gids_vdata,
                    neighboor_src_lids,
                ) = filter_edges(
                    target_gids[i + 1],
                    neighboor_gids_offset,
                    neighboor_gids_vdata,
                    exclude_edge_hashset,
                )
            (
                unique_gids,
                neighbor_raw_to_unique_mapping,
                unique_output_neighbor_count,
            ) = torch.ops.wholegraph.append_unique(
                target_gids[i + 1], neighboor_gids_vdata
            )
            csr_row_ptr[i] = neighboor_gids_offset
            csr_col_ind[i] = neighbor_raw_to_unique_mapping
            sample_dup_count[i] = unique_output_neighbor_count
            neighboor_count = neighboor_gids_vdata.size()[0]
            edge_indice[i] = torch.cat(
                [
                    torch.reshape(neighbor_raw_to_unique_mapping, (1, neighboor_count)),
                    torch.reshape(neighboor_src_lids, (1, neighboor_count)),
                ]
            )
            target_gids[i] = unique_gids
        return target_gids, edge_indice, csr_row_ptr, csr_col_ind, sample_dup_count

    def per_source_negative_sample(
        self, src_nodes: torch.Tensor, negative_sample_count=1
    ):
        is_chunked = isinstance(self.edges_csr_row, wg.ChunkedTensor)
        if is_chunked:
            return torch.ops.wholegraph.per_source_uniform_negative_sample_chunked(
                src_nodes,
                self.edges_csr_row.get_ptr(),
                self.edges_csr_col.get_ptr(),
                self.node_count,
                negative_sample_count,
            )
        else:
            return torch.ops.wholegraph.per_source_uniform_negative_sample(
                src_nodes,
                self.edges_csr_row,
                self.edges_csr_col,
                self.node_count,
                negative_sample_count,
            )

    def create_edges_jump_coo_row(self):
        # jump coo is tensor of ((edge_count)//jump_size, ), each is a src nodeid of edge jump_size * idx
        if self.is_chunked:
            self.edges_jump_coo_row = wg.create_chunked_jump_coo_row(
                self.edges_csr_row, self.edges_csr_col
            )
        else:
            self.edges_jump_coo_row = wg.create_jump_coo_row(
                self.edges_csr_row, self.edges_csr_col, self.use_host_memory
            )

    def prepare_train_edges(self):
        self.start_edge_idx = self.edge_count * comm.get_rank() // comm.get_world_size()
        self.end_edge_idx = (
            self.edge_count * (comm.get_rank() + 1) // comm.get_world_size()
        )
        self.truncate_count = self.edge_count // comm.get_world_size()

    def start_iter(self, batch_size):
        self.batch_size = batch_size
        local_edge_count = self.end_edge_idx - self.start_edge_idx
        selected_count = self.truncate_count // batch_size * batch_size
        self.train_edge_idx_list = (
            torch.randperm(
                local_edge_count, dtype=torch.int64, device="cpu", pin_memory=True
            )
            + self.start_edge_idx
        )
        self.train_edge_idx_list = self.train_edge_idx_list[:selected_count]
        return selected_count // batch_size

    def get_train_edge_batch(self, iter_id):
        start_idx = iter_id * self.batch_size
        end_idx = (iter_id + 1) * self.batch_size
        if self.is_chunked:
            src_nid, dst_nid = wg.get_edge_src_dst_from_eid_chunked(
                self.edges_csr_row,
                self.edges_csr_col,
                self.edges_jump_coo_row,
                self.train_edge_idx_list[start_idx:end_idx].cuda(),
                True,
                True,
            )
        else:
            src_nid, dst_nid = wg.get_edge_src_dst_from_eid(
                self.edges_csr_row,
                self.edges_csr_col,
                self.edges_jump_coo_row,
                self.train_edge_idx_list[start_idx:end_idx].cuda(),
                True,
                True,
            )
        return src_nid, dst_nid

    def gather(self, node_ids, dtype: Union[torch.dtype, None] = None):
        if dtype is None:
            return embedding_ops.EmbeddingLookupFn.apply(node_ids, self.node_feat)
        else:
            return embedding_ops.EmbeddingLookupFn.apply(
                node_ids, self.node_feat, None, dtype
            )


def get_file_names(save_path: str, model_file_prefix: str, idx: int):
    torch_model_file = os.path.join(
        save_path, "_".join([model_file_prefix, "model", str(idx)])
    )
    embedding_dir = os.path.join(
        save_path, "_".join([model_file_prefix, "embeddings", str(idx)])
    )
    return torch_model_file, embedding_dir


def create_node_embedding_meta(graph: HomoGraph):
    meta_dict = {"name": "node_feat"}
    assert len(graph.node_feat.shape) == 2
    embedding_count = graph.node_feat.shape[0]
    embedding_dim = graph.node_feat.shape[1]
    meta_dict["embedding_count"] = embedding_count
    meta_dict["embedding_dim"] = embedding_dim
    meta_dict["part_count"] = wg.get_size(graph.wm_comm)
    embedding_dict = {
        "dtype": pytorch_dtype_to_string(graph.node_feat.embedding.dtype),
        "file_prefix": "embedding",
    }
    meta_dict["embedding"] = embedding_dict
    meta_dict["per_element_states"] = []
    meta_dict["per_embedding_states"] = []
    for i in range(len(graph.node_feat.per_element_states)):
        per_element_state = graph.node_feat.per_element_states[i]
        meta_dict["per_element_states"].append(
            {
                "dtype": pytorch_dtype_to_string(per_element_state.dtype),
                "file_prefix": "_".join(["per_element_state", str(i)]),
            }
        )
    for i in range(len(graph.node_feat.per_embedding_states)):
        per_embedding_state = graph.node_feat.per_embedding_states[i]
        meta_dict["per_embedding_states"].append(
            {
                "dtype": pytorch_dtype_to_string(per_embedding_state.dtype),
                "data_dim": per_embedding_state.shape[1],
                "file_prefix": "_".join(["per_embedding_state", str(i)]),
            }
        )
    return [meta_dict]


def load_node_embedding_meta(graph: HomoGraph, meta_file_path: str):
    import json

    meta_data = json.load(open(meta_file_path, "r"))
    assert len(meta_data) == 1
    meta_dict = meta_data[0]
    assert meta_dict["name"] == "node_feat"
    embedding_count = graph.node_feat.shape[0]
    embedding_dim = graph.node_feat.shape[1]
    assert meta_dict["embedding_count"] == embedding_count
    assert meta_dict["embedding_dim"] == embedding_dim
    embedding_dict = meta_dict["embedding"]
    assert embedding_dict["dtype"] == pytorch_dtype_to_string(
        graph.node_feat.embedding.dtype
    )
    assert embedding_dict["file_prefix"] == "embedding"
    assert len(meta_dict["per_element_states"]) == len(
        graph.node_feat.per_element_states
    )
    for i in range(len(graph.node_feat.per_element_states)):
        per_element_state = graph.node_feat.per_element_states[i]
        per_element_state_dict = meta_dict["per_element_states"][i]
        assert per_element_state_dict["dtype"] == pytorch_dtype_to_string(
            per_element_state.dtype
        )
        assert per_element_state_dict["file_prefix"] == "_".join(
            ["per_element_state", str(i)]
        )
    assert len(meta_dict["per_embedding_states"]) == len(
        graph.node_feat.per_embedding_states
    )
    for i in range(len(graph.node_feat.per_embedding_states)):
        per_embedding_state = graph.node_feat.per_embedding_states[i]
        per_embedding_state_dict = meta_dict["per_embedding_states"][i]
        assert per_embedding_state_dict["dtype"] == pytorch_dtype_to_string(
            per_embedding_state.dtype
        )
        assert per_embedding_state_dict["data_dim"] == per_embedding_state.shape[1]
        assert per_embedding_state_dict["file_prefix"] == "_".join(
            ["per_embedding_state", str(i)]
        )
    return meta_dict


def save_homo_graph_model_state(
    save_path: str,
    model_file_prefix: str,
    model: torch.nn.Module,
    graph: HomoGraph,
    save_idx: int,
):
    torch.distributed.barrier()
    save_file_path, embedding_dir = get_file_names(
        save_path, model_file_prefix, save_idx
    )
    if comm.get_rank() == 0:
        torch.save(model.state_dict(), save_file_path)
        if isinstance(graph.node_feat, embedding_ops.TrainableEmbedding):
            os.mkdir(embedding_dir)
            node_embedding_meta = create_node_embedding_meta(graph)
            import json

            meta_file_path = os.path.join(embedding_dir, "embedding.meta")
            json.dump(node_embedding_meta, open(meta_file_path, "w"))
    torch.distributed.barrier()
    if isinstance(
        graph.node_feat, embedding_ops.TrainableEmbedding
    ) and comm.get_rank() == wg.get_rank(graph.wm_comm):
        part_idx = wg.get_rank(graph.wm_comm)
        lt = embedding_ops.get_local_tensor(graph.node_feat.embedding)
        wg.store_local_tensor_to_embedding_file(
            lt,
            os.path.join(embedding_dir, "_".join(["embedding", "part", str(part_idx)])),
        )
        for i in range(len(graph.node_feat.per_element_states)):
            per_element_state = graph.node_feat.per_element_states[i]
            lt = embedding_ops.get_local_tensor(per_element_state)
            wg.store_local_tensor_to_embedding_file(
                lt,
                os.path.join(
                    embedding_dir,
                    "_".join(["per_element_state", str(i), "part", str(part_idx)]),
                ),
            )
        for i in range(len(graph.node_feat.per_embedding_states)):
            per_embedding_state = graph.node_feat.per_embedding_states[i]
            lt = embedding_ops.get_local_tensor(per_embedding_state)
            wg.store_local_tensor_to_embedding_file(
                lt,
                os.path.join(
                    embedding_dir,
                    "_".join(["per_embedding_state", str(i), "part", str(part_idx)]),
                ),
            )
    torch.distributed.barrier()


def load_homo_graph_model_state(
    save_path: str,
    model_file_prefix: str,
    model: torch.nn.Module,
    graph: HomoGraph,
    load_idx: int = -1,
):
    final_load_idx = load_idx
    if load_idx == -1:
        dir_or_files = os.listdir(save_path)
        max_model_index = -1
        torch_model_prefix = "_".join([model_file_prefix, "model"])
        for dir_file in dir_or_files:
            dir_file_path = os.path.join(save_path, dir_file)
            if not os.path.isdir(dir_file_path):
                if len(dir_file) > len(torch_model_prefix) and dir_file.startswith(
                    torch_model_prefix
                ):
                    str_idx = dir_file[len(torch_model_prefix) + 1 :]
                    if not str_idx.isdigit():
                        continue
                    assert str_idx.isdigit()
                    current_model_index = int(str_idx)
                    if current_model_index > max_model_index:
                        max_model_index = current_model_index
        if max_model_index < 0:
            print(
                "no model parameters with prefix %s found in path %s"
                % (model_file_prefix, save_path)
            )
            raise ValueError("File not found.")
        final_load_idx = max_model_index
    save_file_path, embedding_dir = get_file_names(
        save_path, model_file_prefix, final_load_idx
    )
    model.load_state_dict(torch.load(save_file_path))
    if not isinstance(graph.node_feat, embedding_ops.TrainableEmbedding):
        return
    torch.distributed.barrier()
    meta_file_path = os.path.join(embedding_dir, "embedding.meta")
    meta_dict = load_node_embedding_meta(graph, meta_file_path)
    part_count = meta_dict["part_count"]
    lt = embedding_ops.get_local_tensor(graph.node_feat.embedding)
    wg.load_local_tensor_from_embedding_file(
        lt,
        os.path.join(embedding_dir, "_".join(["embedding", "part"])),
        part_count,
        graph.wm_comm,
    )
    for i in range(len(graph.node_feat.per_element_states)):
        per_element_state = graph.node_feat.per_element_states[i]
        lt = embedding_ops.get_local_tensor(per_element_state)
        wg.load_local_tensor_from_embedding_file(
            lt,
            os.path.join(
                embedding_dir, "_".join(["per_element_state", str(i), "part"])
            ),
            part_count,
            graph.wm_comm,
        )
    for i in range(len(graph.node_feat.per_embedding_states)):
        per_embedding_state = graph.node_feat.per_embedding_states[i]
        lt = embedding_ops.get_local_tensor(per_embedding_state)
        wg.load_local_tensor_from_embedding_file(
            lt,
            os.path.join(
                embedding_dir, "_".join(["per_embedding_state", str(i), "part"])
            ),
            part_count,
            graph.wm_comm,
        )
    torch.distributed.barrier()


def load_pickle_data(
    dataset_dir: str, graph_name: str, is_dataset_root_dir: bool = False
):
    import pickle

    save_dir = dataset_dir
    normalized_graph_name = graph_name_normalize(graph_name)
    if is_dataset_root_dir:
        save_dir = os.path.join(dataset_dir, normalized_graph_name, "converted")
    file_path = os.path.join(save_dir, normalized_graph_name + "_data_and_label.pkl")
    with open(file_path, "rb") as f:
        data_and_label = pickle.load(f)
    train_data = {
        "idx": data_and_label["train_idx"],
        "label": data_and_label["train_label"],
    }
    valid_data = {
        "idx": data_and_label["valid_idx"],
        "label": data_and_label["valid_label"],
    }
    test_data = {
        "idx": data_and_label["test_idx"],
        "label": data_and_label["test_label"],
    }
    return train_data, valid_data, test_data


def load_pickle_link_pred_data(
    dataset_dir: str, graph_name: str, is_dataset_root_dir: bool = False
):
    import pickle

    save_dir = dataset_dir
    normalized_graph_name = graph_name_normalize(graph_name)
    if is_dataset_root_dir:
        save_dir = os.path.join(dataset_dir, normalized_graph_name, "converted")
    file_path = os.path.join(
        save_dir, normalized_graph_name + "_link_prediction_test_valid.pkl"
    )
    with open(file_path, "rb") as f:
        valid_and_test = pickle.load(f)
    return valid_and_test


class NodeClassificationDataset(Dataset):
    def __init__(self, raw_data, global_rank, global_size):
        self.dataset = list(
            list(zip(raw_data["idx"], raw_data["label"].astype(np.int64)))
        )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
