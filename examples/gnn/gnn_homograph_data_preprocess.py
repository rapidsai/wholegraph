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

import os
from optparse import OptionParser

import numpy as np
import torch
from wg_torch.graph_ops import (
    check_data_integrity,
    numpy_dtype_to_string,
    load_meta_file,
    save_meta_file,
    get_part_filename,
    graph_name_normalize,
)

from wholegraph.torch import wholegraph_pytorch as wg


def download_and_convert_node_classification(
    save_dir, ogb_root_dir="dataset", graph_name="ogbn-papers100M"
):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    normalized_graph_name = graph_name_normalize(graph_name)
    support_graph_dict = {
        "ogbn_papers100M": {"node_name": "paper", "relation": "cites"},
        "ogbn_products": {"node_name": "product", "relation": "copurchased"},
    }
    from ogb.nodeproppred import NodePropPredDataset

    assert normalized_graph_name in support_graph_dict.keys()
    node_name = support_graph_dict[normalized_graph_name]["node_name"]
    relation_name = support_graph_dict[normalized_graph_name]["relation"]
    if check_data_integrity(save_dir, normalized_graph_name):
        return
    dataset = NodePropPredDataset(name=graph_name, root=ogb_root_dir)
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
    node_feat_name_prefix = "_".join([normalized_graph_name, "node_feat", node_name])
    edge_index_name_prefix = "_".join(
        [normalized_graph_name, "edge_index", node_name, relation_name, node_name]
    )

    nodes = [
        {
            "name": node_name,
            "has_emb": True,
            "emb_file_prefix": node_feat_name_prefix,
            "num_nodes": num_nodes,
            "emb_dim": node_feat_dim,
            "dtype": numpy_dtype_to_string(node_feat.dtype),
        }
    ]
    edges = [
        {
            "src": node_name,
            "dst": node_name,
            "rel": relation_name,
            "has_emb": False,
            "edge_list_prefix": edge_index_name_prefix,
            "num_edges": num_edges,
            "dtype": numpy_dtype_to_string(np.dtype("int32")),
            "directed": True,
        }
    ]
    meta_json = {"nodes": nodes, "edges": edges}
    save_meta_file(save_dir, meta_json, normalized_graph_name)
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

    with open(
        os.path.join(save_dir, normalized_graph_name + "_data_and_label.pkl"), "wb"
    ) as f:
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


def download_and_convert_link_prediction(
    save_dir, ogb_root_dir="dataset", graph_name="ogbl-citation2"
):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    normalized_graph_name = graph_name_normalize(graph_name)
    support_graph_dict = {"ogbl_citation2": {"node_name": "paper", "relation": "cites"}}
    assert normalized_graph_name in support_graph_dict.keys()
    node_name = support_graph_dict[normalized_graph_name]["node_name"]
    relation_name = support_graph_dict[normalized_graph_name]["relation"]
    if check_data_integrity(save_dir, normalized_graph_name):
        return
    from ogb.linkproppred import LinkPropPredDataset

    dataset = LinkPropPredDataset(name=graph_name, root=ogb_root_dir)
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
    node_feat_name_prefix = "_".join([normalized_graph_name, "node_feat", node_name])
    edge_index_name_prefix = "_".join(
        [normalized_graph_name, "edge_index", node_name, relation_name, node_name]
    )

    nodes = [
        {
            "name": node_name,
            "has_emb": True,
            "emb_file_prefix": node_feat_name_prefix,
            "num_nodes": num_nodes,
            "emb_dim": node_feat_dim,
            "dtype": numpy_dtype_to_string(node_feat.dtype),
        }
    ]
    edges = [
        {
            "src": node_name,
            "dst": node_name,
            "rel": relation_name,
            "has_emb": False,
            "edge_list_prefix": edge_index_name_prefix,
            "num_edges": num_edges,
            "dtype": numpy_dtype_to_string(np.dtype("int32")),
            "directed": True,
        }
    ]
    meta_json = {"nodes": nodes, "edges": edges}
    save_meta_file(save_dir, meta_json, normalized_graph_name)

    valid_test_edges = {"valid": split_edge["valid"], "test": split_edge["test"]}

    train_edges = np.stack(
        (split_edge["train"]["source_node"], split_edge["train"]["target_node"])
    )
    assert (train_edges == edge_index).all()

    import pickle

    with open(
        os.path.join(save_dir, normalized_graph_name + "_node_year.pkl"), "wb"
    ) as f:
        pickle.dump(node_year, f)
    with open(
        os.path.join(
            save_dir, normalized_graph_name + "_link_prediction_test_valid.pkl"
        ),
        "wb",
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


def build_homo_graph(root_dir: str, graph_name: str):
    normalized_graph_name = graph_name_normalize(graph_name)
    output_dir = os.path.join(root_dir, normalized_graph_name, "converted")
    meta_file = load_meta_file(output_dir, normalized_graph_name)
    graph_builder = wg.create_homograph_builder(torch.int32)
    wg.graph_builder_set_shuffle_id(graph_builder, False)
    wg.graph_builder_load_edge_data(
        graph_builder,
        [],
        os.path.join(output_dir, meta_file["edges"][0]["edge_list_prefix"]),
        False,
        torch.int32,
        0,
    )
    wg.graph_builder_set_edge_config(graph_builder, [], True, False, False)
    wg.graph_builder_set_graph_save_file(
        graph_builder,
        os.path.join(output_dir, "homograph_csr_row_ptr"),
        os.path.join(output_dir, "homograph_csr_col_idx"),
        os.path.join(output_dir, "homograph_id_mapping"),
    )

    wg.graph_builder_build(graph_builder)
    wg.destroy_graph_builder(graph_builder)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "-r",
        "--root_dir",
        dest="root_dir",
        default="dataset",
        help="graph root directory.",
    )
    parser.add_option(
        "-g",
        "--graph_name",
        dest="graph_name",
        default="ogbn-papers100M",
        help="graph name, ogbn-papers100M, ogbn-products or ogbl-citation2",
    )
    parser.add_option(
        "-p", "--phase", dest="phase", default="build", help="phase, convert or build"
    )

    (options, args) = parser.parse_args()

    assert options.phase == "convert" or options.phase == "build"

    if options.phase == "convert":
        norm_graph_name = graph_name_normalize(options.graph_name)
        if (
            options.graph_name == "ogbn-papers100M"
            or options.graph_name == "ogbn-products"
        ):
            download_and_convert_node_classification(
                os.path.join(options.root_dir, norm_graph_name, "converted"),
                options.root_dir,
                options.graph_name,
            )
        elif options.graph_name == "ogbl-citation2":
            download_and_convert_link_prediction(
                os.path.join(options.root_dir, norm_graph_name, "converted"),
                options.root_dir,
                options.graph_name,
            )
        else:
            raise ValueError("graph name unknown.")
    else:
        build_homo_graph(os.path.join(options.root_dir), options.graph_name)
