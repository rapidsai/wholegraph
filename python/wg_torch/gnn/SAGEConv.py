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


class SpmmMean(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_neighboor: Tensor,
        csr_row_ptr: Tensor,
        csr_col_ind: Tensor,
        sample_dup_count: Tensor,
    ):
        ctx.save_for_backward(csr_row_ptr, csr_col_ind, sample_dup_count, x_neighboor)
        mean_tensor = torch.ops.wholegraph.spmm_csr_noweight_forward(
            csr_row_ptr, csr_col_ind, x_neighboor, 1
        )
        return mean_tensor

    @staticmethod
    def backward(ctx, grad_outputs: Tensor):
        csr_row_ptr, csr_col_ind, sample_dup_count, x_neighboor = ctx.saved_tensors
        grad_x = torch.ops.wholegraph.spmm_csr_noweight_backward(
            csr_row_ptr, csr_col_ind, sample_dup_count, grad_outputs, 1
        )
        return grad_x, None, None, None


class SpmmGCN(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_neighboor: Tensor,
        csr_row_ptr: Tensor,
        csr_col_ind: Tensor,
        sample_dup_count: Tensor,
    ):
        ctx.save_for_backward(csr_row_ptr, csr_col_ind, sample_dup_count)
        gcn_tensor = torch.ops.wholegraph.spmm_csr_noweight_forward(
            csr_row_ptr, csr_col_ind, x_neighboor, 2
        )
        return gcn_tensor

    @staticmethod
    def backward(ctx, grad_outputs: Tensor):
        csr_row_ptr, csr_col_ind, sample_dup_count = ctx.saved_tensors
        grad_x = torch.ops.wholegraph.spmm_csr_noweight_backward(
            csr_row_ptr, csr_col_ind, sample_dup_count, grad_outputs, 2
        )
        return grad_x, None, None, None


class SAGEConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        root_weight: bool = True,
        bias: bool = True,
        aggregator: str = "mean",
        **kwargs
    ):
        super(SAGEConv, self).__init__()
        if aggregator not in ["mean", "gcn"]:
            raise AssertionError("aggregator %s not supported." % (aggregator,))
        if aggregator == "mean":
            self.aggregator = SpmmMean.apply
        if aggregator == "gcn":
            self.aggregator = SpmmGCN.apply
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.aggregator_type = aggregator

        if aggregator == "mean":
            if isinstance(in_channels, int):
                in_channels = (in_channels, in_channels)
            self.lin = torch.nn.Linear(
                in_channels[0] + in_channels[1], out_channels, bias=bias
            )

        if aggregator == "gcn":
            self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.act = None

        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.lin.weight, gain=gain)

    def forward(
        self,
        csr_row_ptr: Tensor,
        csr_col_ind: Tensor,
        sample_count: Tensor,
        x_neighboor: Tensor,
        x_target: Tensor,
    ):
        x_neighboor_agg = self.aggregator(
            x_neighboor, csr_row_ptr, csr_col_ind, sample_count
        )
        y = x_neighboor_agg

        if self.aggregator_type == "mean":
            y = self.lin(torch.cat([x_target, x_neighboor_agg], 1))
        elif self.aggregator_type == "gcn":
            y = self.lin(x_neighboor_agg)
        if self.bias is not None:
            y = y + self.bias
        if self.act is not None:
            y = self.act(y)
        return y
