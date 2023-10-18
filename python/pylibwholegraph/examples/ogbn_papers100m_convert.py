import argparse
import os
import numpy as np
from scipy.sparse import coo_matrix
import pickle
from ogb.nodeproppred import NodePropPredDataset


def save_array(np_array, save_path, array_file_name):
    array_full_path = os.path.join(save_path, array_file_name)
    with open(array_full_path, 'wb') as f:
        np_array.tofile(f)


def convert_papers100m_dataset(args):
    ogb_root = args.ogb_root_dir
    dataset = NodePropPredDataset(name='ogbn-papers100M', root=ogb_root)
    graph, label = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
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
    num_nodes = graph["num_nodes"]
    edge_index = graph["edge_index"]
    node_feat = graph["node_feat"].astype(np.dtype(args.node_feat_format))
    if not os.path.exists(args.convert_dir):
        print(f"creating directory {args.convert_dir}...")
        os.makedirs(args.convert_dir)
    print("saving idx and labels...")
    with open(
        os.path.join(args.convert_dir, 'ogbn_papers100M_data_and_label.pkl'), "wb"
    ) as f:
        pickle.dump(data_and_label, f)
    print("saving node feature...")
    with open(
            os.path.join(args.convert_dir, 'node_feat.bin'), "wb"
    ) as f:
        node_feat.tofile(f)

    print("converting graph to csr...")
    assert len(edge_index.shape) == 2
    assert edge_index.shape[0] == 2
    coo_src_ids = edge_index[0, :].astype(np.int32)
    coo_dst_ids = edge_index[1, :].astype(np.int32)
    if args.add_reverse_edges:
        arg_graph_src = np.concatenate([coo_src_ids, coo_dst_ids])
        arg_graph_dst = np.concatenate([coo_dst_ids, coo_src_ids])
    else:
        arg_graph_src = coo_src_ids
        arg_graph_dst = coo_dst_ids
    values = np.arange(len(arg_graph_src), dtype='int64')
    coo_graph = coo_matrix((values, (arg_graph_src, arg_graph_dst)), shape=(num_nodes, num_nodes))
    csr_graph = coo_graph.tocsr()
    csr_row_ptr = csr_graph.indptr.astype(dtype='int64')
    csr_col_ind = csr_graph.indices.astype(dtype='int32')
    print("saving csr graph...")
    save_array(csr_row_ptr, args.convert_dir, 'homograph_csr_row_ptr')
    save_array(csr_col_ind, args.convert_dir, 'homograph_csr_col_idx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ogb_root_dir', type=str, default='dataset',
                        help='root dir of containing ogb datasets')
    parser.add_argument('--convert_dir', type=str, default='dataset_papers100m_converted',
                        help='output dir containing converted datasets')
    parser.add_argument('--node_feat_format', type=str, default='float32',
                        choices=['float32', 'float16'],
                        help='save format of node feature')
    parser.add_argument('--add_reverse_edges', type=bool, default=True,
                        help='whether to add reverse edges')
    args = parser.parse_args()
    convert_papers100m_dataset(args)
