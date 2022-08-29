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
import pickle
from optparse import OptionParser

import numpy as np
import torch
import yaml
from ogb.lsc import MAG240MDataset
from wg_torch import graph_ops as graph_ops

from wholegraph.torch import wholegraph_pytorch as wg

meta_file_name = "meta.yaml"


def convert_mag240m_dataset(root_dir: str):
    dir_name = "mag240m_kddcup2021"
    output_dir = os.path.join(root_dir, dir_name, "converted")
    dataset = MAG240MDataset(root=root_dir)
    split_idx = dataset.get_idx_split()
    # print(split_idx)
    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test-whole"],
    )
    edge_index_paper_paper = dataset.edge_index("paper", "paper")
    edge_index_author_paper = dataset.edge_index("author", "paper")
    edge_index_author_institution = dataset.edge_index("author", "institution")
    paper_label = dataset.paper_label

    num_papers = dataset.num_papers
    num_authors = dataset.num_authors
    num_institutions = dataset.num_institutions
    num_paper_features = dataset.num_paper_features
    num_classes = dataset.num_classes

    yaml_dict = {
        "nodes": {
            "paper": {
                "num": num_papers,
                "feature_dim": num_paper_features,
                "feature_dtype": graph_ops.numpy_dtype_to_string(np.dtype("float16")),
            },
            "author": {"num": num_authors, "feature_dim": 0},
            "institution": {"num": num_institutions, "feature_dim": 0},
        },
        "edges": {
            ("paper", "paper", "cites"): {
                "num": edge_index_paper_paper.shape[1],
                "feature_dim": 0,
            },
            ("author", "paper", "writes"): {
                "num": edge_index_author_paper.shape[1],
                "feature_dim": 0,
            },
            ("author", "institution", "affiliated_with"): {
                "num": edge_index_author_institution.shape[1],
                "feature_dim": 0,
            },
        },
        "num_classes": num_classes,
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("saving meta file to %s" % (meta_file_name,))
    with open(os.path.join(output_dir, meta_file_name), "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f)

    # shape (121751666, 768)
    with open(os.path.join(output_dir, "node_feat____paper.bin"), "wb") as f:
        print("saving paper node feat to node_feat____paper.bin")
        dataset.all_paper_feat.tofile(f)

    edges = yaml_dict["edges"]
    for edge in edges:
        edge_index = np.transpose(
            dataset.edge_index(edge[0], edge[2], edge[1]).astype(np.int32)
        )
        edge_idx_filename = os.path.join(
            output_dir,
            "".join(
                ["edge_index____", edge[0], "___", edge[2], "___", edge[1], ".bin"]
            ),
        )
        print(
            "saving (%s--%s->%s) edge index to %s, edge_index (%s)=%s"
            % (
                edge[0],
                edge[2],
                edge[1],
                edge_idx_filename,
                edge_index.shape,
                edge_index,
            )
        )
        with open(edge_idx_filename, "wb") as f:
            edge_index.tofile(f)

    train_label = paper_label[train_idx]
    valid_label = paper_label[valid_idx]
    test_label = paper_label[test_idx]

    print(
        "saving data_and_label to %s"
        % (os.path.join(output_dir, "mag240m_data_and_label.pkl"),)
    )
    data_and_label = {
        "train_idx": train_idx.astype(np.int32),
        "valid_idx": valid_idx.astype(np.int32),
        "test_idx": test_idx.astype(np.int32),
        "train_label": train_label.astype(np.int32),
        "valid_label": valid_label.astype(np.int32),
        "test_label": test_label.astype(np.int32),
    }
    with open(os.path.join(output_dir, "mag240m_data_and_label.pkl"), "wb") as f:
        pickle.dump(data_and_label, f)

    # shape (121751666,)
    with open(os.path.join(output_dir, "paper_year.pkl"), "wb") as f:
        pickle.dump(dataset.all_paper_year, f)


def build_mag240m_mixed_graph(root_dir: str):
    dir_name = "mag240m_kddcup2021"
    output_dir = os.path.join(root_dir, dir_name, "converted")
    with open(os.path.join(output_dir, meta_file_name), "r", encoding="utf-8") as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    node_names = ["paper", "author", "institution"]
    num_papers = yaml_dict["nodes"]["paper"]["num"]
    num_author = yaml_dict["nodes"]["author"]["num"]
    num_institutions = yaml_dict["nodes"]["institution"]["num"]
    relations = [
        ["paper", "cites", "paper"],
        ["author", "writes", "paper"],
        ["author", "affiliated_with", "institution"],
    ]
    graph_builder = wg.create_mixed_graph_builder(node_names, relations, torch.int32)
    wg.graph_builder_set_node_counts(
        graph_builder, [num_papers, num_author, num_institutions]
    )
    for relation in relations:
        wg.graph_builder_load_edge_data(
            graph_builder,
            relation,
            os.path.join(
                output_dir,
                "".join(
                    [
                        "edge_index____",
                        relation[0],
                        "___",
                        relation[1],
                        "___",
                        relation[2],
                        ".bin",
                    ]
                ),
            ),
            False,
            torch.int32,
            0,
        )
    wg.graph_builder_set_edge_config(graph_builder, relations[0], True, False, True)
    wg.graph_builder_set_edge_config(graph_builder, relations[1], False, False, True)
    wg.graph_builder_set_edge_config(graph_builder, relations[2], False, False, True)

    wg.graph_builder_set_graph_save_file(
        graph_builder,
        os.path.join(output_dir, "mixed_graph_csr_row_ptr"),
        os.path.join(output_dir, "mixed_graph_csr_col_idx"),
        os.path.join(output_dir, "mixed_graph_id_mapping"),
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
        "-p", "--phase", dest="phase", default="build", help="phase, convert or build"
    )

    (options, args) = parser.parse_args()

    assert options.phase == "convert" or options.phase == "build"

    if options.phase == "convert":
        convert_mag240m_dataset(options.root_dir)
    else:
        build_mag240m_mixed_graph(options.root_dir)
