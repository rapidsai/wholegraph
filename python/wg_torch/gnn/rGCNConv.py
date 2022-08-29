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

from typing import Union, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter


class SpmmRGCN(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_neighboor: Tensor,
        csr_row_ptr: Tensor,
        csr_col_ind: Tensor,
        edge_type: Tensor,
        sample_dup_count: Tensor,
        num_relation: int,
        aggregator_id: int,
    ):
        ctx.save_for_backward(csr_row_ptr, csr_col_ind, edge_type, sample_dup_count)
        ctx.num_relation = num_relation
        ctx.aggregator_id = aggregator_id
        rgcn_tensor = torch.ops.wholegraph.spmm_csr_relational_noweight_forward(
            csr_row_ptr,
            csr_col_ind,
            edge_type,
            x_neighboor,
            num_relation,
            aggregator_id,
        )
        # [target_count, num_relation * hidden_size]
        return rgcn_tensor

    @staticmethod
    def backward(ctx, grad_outputs: Tensor):
        csr_row_ptr, csr_col_ind, edge_type, sample_dup_count = ctx.saved_tensors
        num_relation = ctx.num_relation
        aggregator_id = ctx.aggregator_id
        grad_x_neighbors = torch.ops.wholegraph.spmm_csr_relational_noweight_backward(
            csr_row_ptr,
            csr_col_ind,
            edge_type,
            sample_dup_count,
            grad_outputs,
            num_relation,
            aggregator_id,
        )
        return grad_x_neighbors, None, None, None, None, None, None


class RGCNConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        root_weight: bool = True,
        bias: bool = True,
        aggregator_type: str = "mean",
    ):
        super(RGCNConv, self).__init__()

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.bias = bias
        self.in_channels_l = in_channels[0]

        self.lin_weight = Parameter(
            torch.empty(out_channels, in_channels[0] * num_relations)
        )

        if root_weight:
            self.root_lin = torch.nn.Linear(in_channels[1], out_channels, bias=bias)
            self.bias_parameter = None
        else:
            self.root_lin = None
            self.bias_parameter = Parameter(torch.zeros(out_channels))
            self.register_parameter("bias", None)
        assert aggregator_type == "mean" or aggregator_type == "sum"
        self.aggregator_id = 0 if aggregator_type == "sum" else 1
        self.aggregator = SpmmRGCN.apply

        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        for i in range(self.num_relations):
            torch.nn.init.xavier_uniform_(self.lin_weight, gain=gain)
        if self.root_lin is not None:
            torch.nn.init.xavier_uniform_(self.root_lin.weight, gain=gain)
            if self.bias_parameter is not None:
                torch.nn.init.zeros_(self.bias_parameter)

    def forward(self, subgraph, x):
        target_id_count = subgraph["target_ids"].shape[0]
        csr_row_ptr = subgraph["csr_row_ptr"]
        csr_col_ind = subgraph["csr_col_ind"]
        dup_count = subgraph["dup_count"]
        edge_type = subgraph["edge_type"]
        num_valid_relation = subgraph["num_relation"]
        x_target = x[:target_id_count]
        x_neighbor = x[target_id_count:]
        x_agg = self.aggregator(
            x_neighbor,
            csr_row_ptr,
            csr_col_ind,
            edge_type,
            dup_count,
            num_valid_relation,
            self.aggregator_id,
        ).view(target_id_count, num_valid_relation * self.in_channels[0])
        y = torch.nn.functional.linear(
            x_agg, self.lin_weight[:, : num_valid_relation * self.in_channels[0]]
        )
        if self.root_lin is not None:
            y += self.root_lin(x_target)
        if self.bias_parameter is not None:
            y += self.bias_parameter
        return y
