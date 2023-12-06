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
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from pylibcugraphops.pytorch.operators import agg_concat_n2n as SAGEConvAgg
from pylibcugraphops.pytorch import SampledCSC


class CuGraphSAGEConv(torch.nn.Module):  # pragma: no cover
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.
    :class:`CuGraphSAGEConv` is an optimized version of
    package that fuses message passing computation for accelerated execution
    and lower memory footprint.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        if aggr not in ["mean", "sum", "min", "max"]:
            raise ValueError(
                f"Aggregation function must be either 'mean', "
                f"'sum', 'min' or 'max' (got '{aggr}')"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if self.project:
            self.pre_lin = Linear(in_channels, in_channels, bias=True)

        if self.root_weight:
            self.lin = Linear(2 * in_channels, out_channels, bias=bias)
        else:
            self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.lin.weight, gain=gain)
        if self.project:
            torch.nn.init.xavier_uniform_(self.pre_lin.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.lin.weight, gain=gain)

    def forward(
        self,
        x: Tensor,
        csr_row_ptr: Tensor,
        csr_col_ind: Tensor,
        max_num_neighbors: int,
    ) -> Tensor:
        graph = SampledCSC(csr_row_ptr, csr_col_ind, max_num_neighbors, x.shape[0])

        if self.project:
            x = self.pre_lin(x).relu()

        out = SAGEConvAgg(x, graph, self.aggr)

        if self.root_weight:
            out = self.lin(out)
        else:
            out = self.lin(out[:, : self.in_channels])

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, aggr={self.aggr})"
        )
