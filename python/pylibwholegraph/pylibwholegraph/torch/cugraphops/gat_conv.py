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
from torch import Tensor
from torch.nn import Linear, Parameter

from pylibcugraphops.pytorch.operators import mha_gat_n2n as GATConvAgg
from pylibcugraphops.pytorch import SampledCSC


class CuGraphGATConv(torch.nn.Module):  # pragma: no cover
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    :class:`CuGraphGATConv` is an optimized version of
    :class:`~torch_geometric.nn.conv.GATConv` based on the :obj:`cugraph-ops`
    package that fuses message passing computation for accelerated execution
    and lower memory footprint.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att = Parameter(torch.Tensor(2 * heads * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_normal_(self.lin.weight, gain=gain)
        torch.nn.init.xavier_normal_(
            self.att.view(2, self.heads, self.out_channels)[0, :, :], gain=gain
        )
        torch.nn.init.xavier_normal_(
            self.att.view(2, self.heads, self.out_channels)[1, :, :], gain=gain
        )
        torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        csr_row_ptr: Tensor,
        csr_col_ind: Tensor,
        max_num_neighbors: int,
    ) -> Tensor:
        graph = SampledCSC(csr_row_ptr, csr_col_ind, max_num_neighbors, x.shape[0])

        x = self.lin(x)

        out = GATConvAgg(
            x,
            self.att,
            graph,
            self.heads,
            "LeakyReLU",
            self.negative_slope,
            self.concat,
        )

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )
