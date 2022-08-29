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

import torch
from torch import Tensor


class gSpmmGAT(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_neighboor: Tensor,
        csr_row_ptr: Tensor,
        csr_col_ind: Tensor,
        edge_weight: Tensor,
        sample_dup_count: Tensor,
    ):
        ctx.save_for_backward(
            x_neighboor, csr_row_ptr, csr_col_ind, edge_weight, sample_dup_count
        )
        out_tensor = torch.ops.wholegraph.gspmm_csr_weighted_forward(
            csr_row_ptr, csr_col_ind, x_neighboor, edge_weight
        )
        return out_tensor

    @staticmethod
    def backward(ctx, grad_outputs: Tensor):
        (
            x_neighboor,
            csr_row_ptr,
            csr_col_ind,
            edge_weight,
            sample_dup_count,
        ) = ctx.saved_tensors
        x_neighboor_grad = None
        edge_weight_grad = None
        in_grads = torch.ops.wholegraph.gspmm_csr_weighted_backward(
            csr_row_ptr,
            csr_col_ind,
            x_neighboor,
            edge_weight,
            sample_dup_count,
            grad_outputs,
        )
        if x_neighboor.requires_grad and edge_weight.requires_grad:
            x_neighboor_grad, edge_weight_grad = in_grads
        elif x_neighboor.requires_grad:
            x_neighboor_grad = in_grads
        else:
            edge_weight_grad = in_grads

        return x_neighboor_grad, None, None, edge_weight_grad, None


class SpAddGAT(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        edge_weight_left: Tensor,
        edge_weight_right: Tensor,
        csr_row_ptr: Tensor,
        csr_col_ind: Tensor,
        sample_dup_count: Tensor,
    ):
        ctx.save_for_backward(csr_row_ptr, csr_col_ind, sample_dup_count)
        weight_graph_val_tensor = torch.ops.wholegraph.spadd_gat_csr_forward(
            csr_row_ptr, csr_col_ind, edge_weight_left, edge_weight_right
        )
        return weight_graph_val_tensor

    @staticmethod
    def backward(ctx, grad_outputs: Tensor):
        csr_row_ptr, csr_col_ind, sample_dup_count = ctx.saved_tensors
        (
            grad_weight_left,
            grad_weight_right,
        ) = torch.ops.wholegraph.spadd_gat_csr_backward(
            csr_row_ptr, csr_col_ind, sample_dup_count, grad_outputs
        )
        return grad_weight_left, grad_weight_right, None, None, None


class SpSoftMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_weight: Tensor, csr_row_ptr: Tensor):
        edge_weight_softmax = torch.ops.wholegraph.edge_weight_softmax_forward(
            csr_row_ptr, edge_weight
        )
        ctx.save_for_backward(csr_row_ptr, edge_weight)
        return edge_weight_softmax

    @staticmethod
    def backward(ctx, grad_outputs: Tensor):
        csr_row_ptr, edge_weight = ctx.saved_tensors
        grad_weight_softmax = torch.ops.wholegraph.edge_weight_softmax_backward(
            csr_row_ptr, edge_weight, grad_outputs
        )
        return grad_weight_softmax, None


class GATConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 1,
        bias: bool = True,
        negative_slope: float = 0.2,
        add_self_loop: bool = True,
        mean_output: bool = False,
        **kwargs
    ):
        super(GATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.add_self_loop = add_self_loop
        self.mean_output = mean_output
        self.lin = torch.nn.Linear(in_channels, out_channels * num_heads, bias=False)
        # left for target
        self.attn_l = torch.nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_channels))
        )
        # right for neighbor
        self.attn_r = torch.nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_channels))
        )
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(size=(num_heads * out_channels,))
            )
        else:
            self.bias = None
            # self.register_buffer('bias', None)

        self.aggregator = gSpmmGAT.apply
        self.attention_compute = SpAddGAT.apply
        self.edge_softmax = SpSoftMax.apply

        self.act = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_normal_(self.lin.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.attn_l, gain=gain)
        torch.nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(
        self,
        csr_row_ptr: Tensor,
        csr_col_ind: Tensor,
        sample_count: Tensor,
        x_neighboor: Tensor,
        x_target: Tensor,
    ):
        if self.add_self_loop:
            (
                csr_row_ptr_looped,
                csr_col_ind_looped,
                sample_count_looped,
            ) = torch.ops.wholegraph.csr_add_self_loop(
                csr_row_ptr, csr_col_ind, sample_count
            )
        else:
            csr_row_ptr_looped, csr_col_ind_looped, sample_count_looped = (
                csr_row_ptr,
                csr_col_ind,
                sample_count,
            )

        target_node_num = csr_row_ptr_looped.shape[0] - 1
        target_prefix_shape = neighbor_prefix_shape = x_neighboor.shape[:-1]
        target_prefix_shape = (target_node_num,) + target_prefix_shape[1:]

        # (N, d_in) -> (N, d_out*num_heads)
        # print('x_neighboor.shape=%s, lin.weight.shape=%s' % (str(x_neighboor.shape), str(self.lin.weight.shape)))
        h_neighbor = self.lin(x_neighboor)

        # (N, d_out*num_heads) -> (N, num_heads, d_out)
        feat_neighbor = h_neighbor.view(
            *neighbor_prefix_shape, self.num_heads, self.out_channels
        )
        feat_target = feat_neighbor[:target_node_num]

        # ->(TN, num_heads)
        edge_weight_left = (feat_target * self.attn_l).sum(dim=-1)
        # edge_weight_left = torch.einsum('ijk,jk->ij', feat_target, self.attn_l.view(self.num_heads, self.out_channels))
        # ->(N, num_heads)
        edge_weight_right = (feat_neighbor * self.attn_r).sum(dim=-1)
        # edge_weight_right = torch.einsum('ijk,jk->ij', feat_neighbor, self.attn_r.view(self.num_heads, self.out_channels))

        # -> (edge_num, num_heads)
        # print('edge_weight_left[%s]=%s\n' % (str(edge_weight_left.shape), edge_weight_left))
        # print('edge_weight_right[%s]=%s\n' % (str(edge_weight_right.shape), edge_weight_right))
        csr_weight_graph_val = self.attention_compute(
            edge_weight_left,
            edge_weight_right,
            csr_row_ptr_looped,
            csr_col_ind_looped,
            sample_count_looped,
        )

        # compute leakyrelu
        edge_weight = self.leaky_relu(csr_weight_graph_val)

        # print('edge_weight[%s]=%s\n' % (str(edge_weight.shape), edge_weight))

        # softmax edges
        edge_weight = self.edge_softmax(edge_weight, csr_row_ptr_looped)

        # -> (TN, num_heads, d_out)
        y = self.aggregator(
            feat_neighbor,
            csr_row_ptr_looped,
            csr_col_ind_looped,
            edge_weight,
            sample_count_looped,
        )

        if self.bias is not None:
            y = y + self.bias.view(
                *((1,) * len(target_prefix_shape)), self.num_heads, self.out_channels
            )

        if self.act is not None:
            y = self.act(y)

        if self.mean_output:
            y = y.mean(dim=1)
        else:
            y = y.flatten(start_dim=1)
        return y
