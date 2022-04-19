# graph related operations

import os
import re
from typing import Union

import numpy as np
import torch
from wholememory.torch import wholememory_pytorch as wm
from torch.utils.data import Dataset, DataLoader


def load_meta_file(save_dir, graph_name):
    meta_file_name = graph_name + '_meta.json'
    meta_file_path = os.path.join(save_dir, meta_file_name)
    import json
    meta_data = json.load(open(meta_file_path, 'r'))
    return meta_data


def save_meta_file(save_dir, meta_data, graph_name):
    meta_file_name = graph_name + '_meta.json'
    meta_file_path = os.path.join(save_dir, meta_file_name)
    import json
    json.dump(meta_data, open(meta_file_path, 'w'))


numpy_dtype_to_string_dict = {np.dtype('float16'): 'half', np.dtype('float32'): 'float32', np.dtype('float64'): 'double', np.dtype('int8'): 'int8',
                              np.dtype('int16'): 'int16', np.dtype('int32'): 'int32', np.dtype('int64'): 'int64'}
string_to_pytorch_dtype_dict = {'half': torch.float16, 'float32': torch.float32, 'double': torch.float64, 'int8': torch.int8,
                        'int16': torch.int16, 'int32': torch.int32, 'int64': torch.int64}


def numpy_dtype_to_string(dtype: torch.dtype):
    if dtype not in numpy_dtype_to_string_dict.keys():
        print('dtype type %s %s not in dict' % (type(dtype), dtype))
        raise ValueError
    return numpy_dtype_to_string_dict[dtype]


def string_to_pytorch_dtype(dtype_str: str):
    if dtype_str not in string_to_pytorch_dtype_dict.keys():
        print('string dtype %s not in dict' % (dtype_str, ))
        raise ValueError
    return string_to_pytorch_dtype_dict[dtype_str]


def parse_part_file(part_file_name: str, prefix: str):
    if not part_file_name.startswith(prefix):
        return None, None
    if part_file_name == prefix:
        return 0, 1
    pattern = re.compile('_part_(\d+)_of_(\d+)')
    matches = pattern.match(part_file_name[len(prefix):])
    int_tuple = matches.groups()
    if len(int_tuple) != 2:
        return None, None
    return int(int_tuple[0]), int(int_tuple[1])


def get_part_filename(prefix: str, idx: int = 0, count: int = 1):
    filename = prefix
    filename += '_part_%d_of_%d' % (idx, count)
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
            raise FileExistsError('prefix %s both count=%d and count=%d exist.' % (prefix, total_file_count, count))
    if valid_files == total_file_count:
        return total_file_count
    if total_file_count != valid_files:
        raise FileNotFoundError('prefix %s count=%d but got only %d files.' % (prefix, total_file_count, valid_files))
    return None


def check_data_integrity(save_dir, graph_name):
    meta_file_name = graph_name + '_meta.json'
    meta_file_path = os.path.join(save_dir, meta_file_name)
    if not os.path.exists(meta_file_path):
        return False
    if not os.path.isfile(meta_file_path):
        return False
    meta_data = load_meta_file(save_dir, graph_name)
    if meta_data is None:
        return False

    for node_type in meta_data['nodes']:
        if node_type['has_emb']:
            node_emb_prefix = node_type['emb_file_prefix']
            emb_file_count = check_part_files_in_path(save_dir, node_emb_prefix)
            if emb_file_count == 0:
                return False

    for edge_type in meta_data['edges']:
        edge_list_prefix = edge_type['edge_list_prefix']
        edge_file_count = check_part_files_in_path(save_dir, edge_list_prefix)
        if edge_file_count == 0:
            return False
        if edge_type['has_emb']:
            edge_emb_prefix = edge_type['emb_file_prefix']
            emb_file_count = check_part_files_in_path(save_dir, edge_emb_prefix)
            if emb_file_count == 0:
                return False

    return True


def download_and_convert_papers100m(save_dir, ogb_root_dir="dataset"):
    graph_name = 'papers100m'
    from ogb.nodeproppred import NodePropPredDataset
    if check_data_integrity(save_dir, graph_name):
        return
    dataset = NodePropPredDataset(name='ogbn-papers100M', root=ogb_root_dir)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0]
    for name in ['num_nodes', 'edge_index', 'node_feat', 'edge_feat']:
        if name not in graph.keys():
            raise ValueError('graph has no key %s, graph.keys()= %s' % (name, graph.keys()))
    num_nodes = graph['num_nodes']
    edge_index = graph['edge_index']
    node_feat = graph['node_feat']
    edge_feat = graph['edge_feat']
    if isinstance(num_nodes, np.int64) or isinstance(num_nodes, np.int32):
        num_nodes = num_nodes.item()
    if not isinstance(edge_index, np.ndarray) or len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
        raise TypeError('edge_index is not numpy.ndarray of shape (2, x)')
    num_edges = edge_index.shape[1]
    assert node_feat is not None
    if not isinstance(node_feat, np.ndarray) or len(node_feat.shape) != 2 or node_feat.shape[0] != num_nodes:
        raise ValueError('node_feat is not numpy.ndarray of shape (num_nodes, x)')
    node_feat_dim = node_feat.shape[1]
    node_feat_name_prefix = 'papers100m_node_feat_paper'
    edge_index_name_prefix = 'papers100m_edge_index_paper_cites_paper'
    '''
    edge_feat_dim = 0
    if edge_feat is not None:
        if not isinstance(edge_feat, np.ndarray) or len(edge_feat.shape) != 2 or edge_feat.shape[0] != num_edges:
            raise ValueError('edge_feat is not numpy.ndarray of shape (num_edges, x)')
        edge_feat_dim = edge_feat.shape[1]
    '''
    nodes = [{'name': 'paper', 'has_emb': True, 'emb_file_prefix': node_feat_name_prefix, 'num_nodes': num_nodes,
              'emb_dim': node_feat_dim, 'dtype': numpy_dtype_to_string(node_feat.dtype)}]
    edges = [{'src': 'paper', 'dst': 'paper', 'rel': 'cites', 'has_emb': False, 'edge_list_prefix': edge_index_name_prefix,
              'num_edges': num_edges, 'dtype': numpy_dtype_to_string(np.dtype('int32')), 'directed': True}]
    meta_json = {'nodes': nodes, 'edges': edges}
    save_meta_file(save_dir, meta_json, graph_name)
    train_label = label[train_idx]
    valid_label = label[valid_idx]
    test_label = label[test_idx]
    data_and_label = {'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': test_idx,
                      'train_label': train_label, 'valid_label': valid_label, 'test_label': test_label}
    import pickle
    with open(os.path.join(save_dir, graph_name + '_data_and_label.pkl'), "wb") as f:
        pickle.dump(data_and_label, f)
    print('saving node feature...')
    with open(os.path.join(save_dir, get_part_filename(node_feat_name_prefix)), "wb") as f:
        node_feat.tofile(f)
    print('converting edge index...')
    edge_index_int32 = np.transpose(edge_index).astype(np.int32)
    print('saving edge index...')
    with open(os.path.join(save_dir, get_part_filename(edge_index_name_prefix)), "wb") as f:
        edge_index_int32.tofile(f)

    assert edge_feat is None


def allocate_wholememory_tensor(sizes, strides, dtype, use_chunked: bool, use_host_mem: bool=False):
    if use_chunked:
        assert use_host_mem == False
        return wm.create_chunked_tensor(sizes, strides, dtype, [])
    else:
        return wm.create_tensor(sizes, strides, dtype, use_host_mem, [])


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
        self.node_info = None
        self.edge_info = None

    def id_type(self):
        return self.id_dtype

    def node_feat_dtype(self):
        return self.feat_dtype

    def node_feat_shape(self):
        return self.node_feat.shape

    def load(self, save_dir: str, graph_name: str, use_chunked: bool, use_host_memory: bool = False,
             feat_dtype: Union[torch.dtype, None] = None, id_dtype: Union[torch.dtype, None] = None):
        self.is_chunked = use_chunked
        if not check_data_integrity(save_dir, graph_name):
            print('path %s doesn\'t contain all the data for %s' % (save_dir, graph_name))
            raise FileNotFoundError
        self.meta_data = load_meta_file(save_dir, graph_name)
        nodes = self.meta_data['nodes']
        edges = self.meta_data['edges']
        assert len(nodes) == 1
        assert len(edges) == 1
        self.node_info = nodes[0]
        self.edge_info = edges[0]
        self.node_count = nodes[0]['num_nodes']
        data_edge_count = edges[0]['num_edges']

        if id_dtype is None:
            id_dtype = string_to_pytorch_dtype(edges[0]['dtype'])
        self.id_dtype = id_dtype

        edge_index_file_prefix = os.path.join(save_dir, edges[0]['edge_list_prefix'])
        edge_list = wm.load_edge_index_from_file_prefix(edge_index_file_prefix, id_dtype, 0)
        if use_chunked:
            self.edges_csr_row, self.edges_csr_col = wm.allocate_chunked_csr_graph(edge_list, id_dtype, False, False, False)
            self.edge_count = wm.load_to_chunked_csr_graph_from_edge_buffer(edge_list, self.edges_csr_row, self.edges_csr_col,
                                                                            id_dtype, False, False, False)
            self.edges_csr_col = wm.get_sub_chunked_tensor(self.edges_csr_col, [0], [self.edge_count])
        else:
            self.edges_csr_row, self.edges_csr_col = wm.allocate_csr_graph(edge_list, id_dtype, False, False, False, use_host_memory)
            self.edge_count = wm.load_to_csr_graph_from_edge_buffer(edge_list, self.edges_csr_row, self.edges_csr_col,
                                                                    id_dtype, False, False, False)
            self.edges_csr_col = self.edges_csr_col[:self.edge_count]
        wm.free_edge_index(edge_list)

        if (nodes[0]['has_emb']):
            embedding_dim = nodes[0]['emb_dim']
            self.node_feat = allocate_wholememory_tensor([self.node_count, embedding_dim], [], torch.float32, use_chunked, use_host_memory)
            src_dtype = string_to_pytorch_dtype(nodes[0]['dtype'])
            if feat_dtype is None:
                feat_dtype = src_dtype
            self.feat_dtype = feat_dtype
            node_emb_file_prefix = os.path.join(save_dir, nodes[0]['emb_file_prefix'])
            if use_chunked:
                wm.load_embedding_to_chunked_tensor_from_file_prefix(self.node_feat, src_dtype, feat_dtype, self.node_count, embedding_dim, node_emb_file_prefix)
            else:
                wm.load_embedding_to_tensor_from_file_prefix(self.node_feat, src_dtype, feat_dtype, self.node_count, embedding_dim, node_emb_file_prefix)

    def unweighted_sample_without_replacement(self, node_ids, max_neighbors):
        hops = len(max_neighbors)
        sample_dup_count = [None] * hops
        edge_indice = [None] * hops
        csr_row_ptr = [None] * hops
        csr_col_ind = [None] * hops
        target_gids = [None] * (hops + 1)
        target_gids[hops] = node_ids
        for i in range(hops - 1, -1, -1):
            if self.is_chunked:
                neighboor_gids_offset, neighboor_gids_vdata, neighboor_src_lids = torch.ops.wholememory.unweighted_sample_without_replacement_chunked(
                    target_gids[i + 1], self.edges_csr_row.get_ptr(), self.edges_csr_col.get_ptr(), max_neighbors[hops - i - 1])
            else:
                neighboor_gids_offset, neighboor_gids_vdata, neighboor_src_lids = torch.ops.wholememory.unweighted_sample_without_replacement(
                    target_gids[i + 1], self.edges_csr_row, self.edges_csr_col, max_neighbors[hops - i - 1])
            unique_gids, neighbor_raw_to_unique_mapping, unique_output_neighbor_count = torch.ops.wholememory.append_unique(
                target_gids[i + 1], neighboor_gids_vdata)
            csr_row_ptr[i] = neighboor_gids_offset
            csr_col_ind[i] = neighbor_raw_to_unique_mapping
            sample_dup_count[i] = unique_output_neighbor_count
            neighboor_count = neighboor_gids_vdata.size()[0]
            edge_indice[i] = torch.cat([torch.reshape(neighbor_raw_to_unique_mapping, (1, neighboor_count)),
                                        torch.reshape(neighboor_src_lids, (1, neighboor_count))])
            target_gids[i] = unique_gids
        return target_gids, edge_indice, csr_row_ptr, csr_col_ind, sample_dup_count

    def gather(self, node_ids):
        if self.is_chunked:
            return torch.ops.wholememory.gather_chunked(node_ids, self.node_feat.get_ptr(), self.node_feat_dtype())
        else:
            return torch.ops.wholememory.gather(node_ids, self.node_feat, self.node_feat_dtype())


def load_pickle_data(save_dir: str, graph_name: str):
    import pickle
    file_path = os.path.join(save_dir, graph_name + '_data_and_label.pkl')
    with open(file_path, "rb") as f:
        data_and_label = pickle.load(f)
    train_data = {'idx': data_and_label['train_idx'], 'label': data_and_label['train_label']}
    valid_data = {'idx': data_and_label['valid_idx'], 'label': data_and_label['valid_label']}
    test_data = {'idx': data_and_label['test_idx'], 'label': data_and_label['test_label']}
    return train_data, valid_data, test_data


class NodeClassificationDataset(Dataset):
    def __init__(self, raw_data, global_rank, global_size):
        self.dataset = list(list(zip(raw_data['idx'], raw_data['label'].astype(np.int64))))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

