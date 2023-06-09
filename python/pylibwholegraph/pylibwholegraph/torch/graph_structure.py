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
from typing import Union, List
from .tensor import WholeMemoryTensor
from . import graph_ops
from . import wholegraph_ops


class GraphStructure(object):
    r"""Graph structure storage
    Actually, it is the graph structure of one relation, represented in CSR format.
    """

    def __init__(self):
        super().__init__()
        self.node_count = 0
        self.edge_count = 0
        self.csr_row_ptr = None
        self.csr_col_ind = None
        self.node_attributes = {}
        self.edge_attributes = {}

    def set_csr_graph(
        self, csr_row_ptr: WholeMemoryTensor, csr_col_ind: WholeMemoryTensor
    ):
        assert csr_row_ptr.dim() == 1
        assert csr_row_ptr.dtype == torch.int64
        assert csr_row_ptr.shape[0] > 1
        self.node_count = csr_row_ptr.shape[0] - 1
        self.edge_count = csr_col_ind.shape[0]
        assert csr_col_ind.dim() == 1
        assert csr_col_ind.dtype == torch.int32 or csr_col_ind.dtype == torch.int64
        self.csr_row_ptr = csr_row_ptr
        self.csr_col_ind = csr_col_ind

    def set_node_attribute(self, attr_name: str, attr_tensor: WholeMemoryTensor):
        assert attr_name not in self.node_attributes
        assert attr_tensor.shape[0] == self.node_count
        self.node_attributes[attr_name] = attr_tensor

    def set_edge_attribute(self, attr_name: str, attr_tensor: WholeMemoryTensor):
        assert attr_name not in self.edge_attributes
        assert attr_tensor.shape[0] == self.edge_count
        self.edge_attributes[attr_name] = attr_tensor

    def unweighted_sample_without_replacement_one_hop(
        self,
        centor_nodes_tensor: torch.Tensor,
        max_sample_count: int,
        *,
        random_seed: Union[int, None] = None,
        need_center_local_output: bool = False,
        need_edge_output: bool = False
    ):
        return wholegraph_ops.unweighted_sample_without_replacement(
            self.csr_row_ptr.wmb_tensor,
            self.csr_col_ind.wmb_tensor,
            centor_nodes_tensor,
            max_sample_count,
            random_seed,
            need_center_local_output,
            need_edge_output,
        )

    def weighted_sample_without_replacement_one_hop(
        self,
        weight_name: str,
        center_nodes_tensor: torch.Tensor,
        max_sample_count: int,
        *,
        random_seed: Union[int, None] = None,
        need_center_local_output: bool = False,
        need_edge_output: bool = False
    ):
        assert weight_name in self.edge_attributes
        weight_tensor = self.edge_attributes[weight_name]
        return wholegraph_ops.weighted_sample_without_replacement(
            self.csr_row_ptr.wmb_tensor,
            self.csr_col_ind.wmb_tensor,
            weight_tensor.wmb_tensor,
            center_nodes_tensor,
            max_sample_count,
            random_seed,
            need_center_local_output,
            need_edge_output,
        )

    def multilayer_sample_without_replacement(
        self,
        node_ids: torch.Tensor,
        max_neighbors: List[int],
        weight_name: Union[str, None] = None,
    ):
        hops = len(max_neighbors)
        edge_indice = [None] * hops
        csr_row_ptr = [None] * hops
        csr_col_ind = [None] * hops
        target_gids = [None] * (hops + 1)
        target_gids[hops] = node_ids
        for i in range(hops - 1, -1, -1):
            if weight_name is None:
                (
                    neighbor_gids_offset,
                    neighbor_gids_vdata,
                    neighbor_src_lids,
                ) = self.unweighted_sample_without_replacement_one_hop(
                    target_gids[i + 1],
                    max_neighbors[hops - i - 1],
                    need_center_local_output=True,
                )
            else:
                (
                    neighbor_gids_offset,
                    neighbor_gids_vdata,
                    neighbor_src_lids,
                ) = self.weighted_sample_without_replacement_one_hop(
                    weight_name,
                    target_gids[i + 1],
                    max_neighbors[hops - i - 1],
                    need_center_local_output=True,
                )
            (unique_gids, neighbor_raw_to_unique_mapping,) = graph_ops.append_unique(
                target_gids[i + 1],
                neighbor_gids_vdata,
                need_neighbor_raw_to_unique=True,
            )
            csr_row_ptr[i] = neighbor_gids_offset
            csr_col_ind[i] = neighbor_raw_to_unique_mapping
            neighbor_count = neighbor_gids_vdata.size()[0]
            edge_indice[i] = torch.cat(
                [
                    torch.reshape(neighbor_raw_to_unique_mapping, (1, neighbor_count)),
                    torch.reshape(neighbor_src_lids, (1, neighbor_count)),
                ]
            )
            target_gids[i] = unique_gids
        return target_gids, edge_indice, csr_row_ptr, csr_col_ind
