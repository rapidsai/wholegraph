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

import time
from typing import Tuple

import numpy as np
import torch
from wholegraph.torch import wholegraph_pytorch as wg


def gen_input_of_single_unweighted_sampler_instance(neighbor_count: int):
    input_nodes = torch.tensor([0], dtype=torch.int64).cuda()
    csr_row_ptr = torch.tensor([0, neighbor_count], dtype=torch.int64).cuda()
    csr_col_ind = torch.arange(0, neighbor_count, dtype=torch.int64).cuda()

    return input_nodes, csr_row_ptr, csr_col_ind


def gen_input_of_single_weighted_sampler_instance_with_uniform_weight(
    neighbor_count: int,
):
    input_nodes = torch.tensor([0], dtype=torch.int64).cuda()
    csr_row_ptr = torch.tensor([0, neighbor_count], dtype=torch.int64).cuda()
    csr_col_ind = torch.arange(0, neighbor_count, dtype=torch.int64).cuda()
    csr_weight_ptr = torch.ones(neighbor_count, dtype=torch.float32).cuda()
    # csr_weight_ptr=torch.randn(neighbor_count,dtype=torch.float32).cuda()

    return input_nodes, csr_row_ptr, csr_col_ind, csr_weight_ptr


def gen_input_of_single_weighted_sampler_instance_with_random_weight(
    neighbor_count: int,
):
    input_nodes = torch.tensor([0], dtype=torch.int64).cuda()
    csr_row_ptr = torch.tensor([0, neighbor_count], dtype=torch.int64).cuda()
    csr_col_ind = torch.arange(0, neighbor_count, dtype=torch.int64).cuda()
    csr_weight_ptr = torch.rand(neighbor_count, dtype=torch.float32).cuda()
    # csr_weight_ptr=torch.randn(neighbor_count,dtype=torch.float32).cuda()
    return input_nodes, csr_row_ptr, csr_col_ind, csr_weight_ptr


def gen_hist_with_lambda(max_iter, neighbor_count, sample_fun):
    hist = torch.zeros(neighbor_count).cuda()
    for iter in range(max_iter):
        sample = sample_fun()
        assert sample.unique().size() == sample.size()
        # sample_to_array= torch.zeros(neighbor_count).cuda()
        # sample_to_array[sample]=1
        # hist=hist+sample_to_array
        hist[sample] += 1
    return hist


def cal_cdf_from_hist(hist: torch.Tensor) -> torch.Tensor:
    prefix_sum: torch.Tensor = hist.cumsum(dim=0)
    cdf = prefix_sum / prefix_sum[-1]
    return cdf


def check_two_sample_hist(
    max_iter, expect_hist: torch.Tensor, actual_hist: torch.Tensor, threshold=0.9
):
    expect_cdf = cal_cdf_from_hist(expect_hist)
    actual_cdf = cal_cdf_from_hist(actual_hist)
    ks_stat = (expect_cdf - actual_cdf).abs().max().cpu()

    expect_size = expect_hist.size(0)
    m = expect_size
    actual_size = actual_hist.size(0)
    n = actual_size
    en = np.round(expect_size * actual_size / (expect_size + actual_size))
    # p_value = scipy.stats.distributions.kstwo.sf(ks_stat,en)
    # stats._attempt_exact_2kssamp()
    # ks_stat, p_value = stats.kstest(actual_hist.cpu(),expect_hist.cpu())
    #  Similar to ks.test in R
    z = np.sqrt(en) * ks_stat
    expt = -2 * z**2 - 2 * z * (m + 2 * n) / np.sqrt(m * n * (m + n)) / 3.0
    p_value = np.exp(expt)
    print(" ks_stat {} p_value {} ".format(ks_stat, p_value))
    assert p_value > threshold


def test_single_weighted_sample_without_replacement_with_uniform_weight(
    max_sample_count, neighbor_count, max_iter
):
    (
        input_nodes,
        csr_row_ptr,
        csr_col_ind,
        csr_weight_ptr,
    ) = gen_input_of_single_weighted_sampler_instance_with_uniform_weight(
        neighbor_count
    )
    expect_fun = lambda: torch.ops.wholegraph.unweighted_sample_without_replacement(
        input_nodes, csr_row_ptr, csr_col_ind, max_sample_count
    )[1]
    actual_fun = lambda: torch.ops.wholegraph.weighted_sample_without_replacement(
        input_nodes, csr_row_ptr, csr_col_ind, csr_weight_ptr, max_sample_count, None
    )[1]
    expect_hist = gen_hist_with_lambda(max_iter, neighbor_count, expect_fun)
    actual_hist = gen_hist_with_lambda(max_iter, neighbor_count, actual_fun)
    check_two_sample_hist(max_iter, expect_hist, actual_hist)


def test_single_weighted_sample_without_replacement_with_random_weight(
    max_sample_count, neighbor_count, max_iter
):
    (
        input_nodes,
        csr_row_ptr,
        csr_col_ind,
        csr_weight_ptr,
    ) = gen_input_of_single_weighted_sampler_instance_with_random_weight(neighbor_count)
    expect_fun = lambda: torch.multinomial(
        csr_weight_ptr, num_samples=max_sample_count, replacement=False
    )
    # expect_fun = lambda : torch.ops.wholegraph.unweighted_sample_without_replacement(input_nodes,csr_row_ptr,csr_col_ind,max_sample_count)[1]
    actual_fun = lambda: torch.ops.wholegraph.weighted_sample_without_replacement(
        input_nodes, csr_row_ptr, csr_col_ind, csr_weight_ptr, max_sample_count, None
    )[1]
    start_time = time.time()
    expect_hist = gen_hist_with_lambda(max_iter, neighbor_count, expect_fun)
    end_time = time.time()
    print(" torch time {} ms ".format(end_time - start_time))
    start_time = time.time()

    actual_hist = gen_hist_with_lambda(max_iter, neighbor_count, actual_fun)
    print(" wholegraph time {} ms ".format(time.time() - start_time))

    check_two_sample_hist(max_iter, expect_hist, actual_hist)


def create_random_csr_graph_and_target_nodes(
    num_nodes: int, num_edges: int, target_nodes_num: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_count = num_nodes * num_nodes
    matrix_tensor = torch.rand(
        all_count, dtype=torch.float32, device=torch.device("cpu")
    )
    choice_zero_idxs = torch.randperm(all_count, device=torch.device("cpu"))[
        : all_count - num_edges
    ]
    matrix_tensor[choice_zero_idxs] = 0
    matrix_tensor.resize_(num_nodes, num_nodes)
    target_node_tensor = torch.randint(
        0,
        num_nodes,
        (target_nodes_num,),
        dtype=torch.int64,
        device=torch.device("cuda"),
    ).unique()
    print(" target_node  count is {} *******".format(target_node_tensor.size(0)))
    sp_format = matrix_tensor.to_sparse_csr()

    return (
        sp_format.crow_indices().cuda(),
        sp_format.col_indices().cuda(),
        sp_format.values().cuda(),
        target_node_tensor,
    )


def check_function_of_weighted_sample(
    max_iter: int,
    max_sample_count: int,
    input_nodes: torch.Tensor,
    csr_row_ptr: torch.Tensor,
    csr_col_ind: torch.Tensor,
    csr_weight_ptr: torch.Tensor,
):
    (
        sample_offset_tensor,
        sample_output,
        center_localid,
    ) = torch.ops.wholegraph.weighted_sample_without_replacement(
        input_nodes, csr_row_ptr, csr_col_ind, csr_weight_ptr, max_sample_count, None
    )
    check_node_idx = torch.randint(0, input_nodes.size(0), (1,))

    check_node = input_nodes[check_node_idx]
    neighbor_start = csr_row_ptr[check_node]
    neighbor_end = csr_row_ptr[check_node + 1]
    check_neighbor = csr_col_ind[neighbor_start:neighbor_end]
    check_weights = csr_weight_ptr[neighbor_start:neighbor_end]

    sample_offset_start = sample_offset_tensor[check_node_idx]
    sample_offset_end = sample_offset_tensor[check_node_idx + 1]
    check_center_localids = center_localid[sample_offset_start:sample_offset_end]
    assert check_center_localids.unique().size(0) == 1
    assert check_center_localids[0].cpu() == check_node_idx
    assert check_neighbor.unique().size(0) == check_neighbor.size(0)
    expect_fun = lambda: check_neighbor[
        torch.multinomial(
            check_weights, num_samples=max_sample_count, replacement=False
        )
    ]

    def actual_fun():
        (
            local_sample_offset_tensor,
            local_sample_output,
            local_center_localid,
        ) = torch.ops.wholegraph.weighted_sample_without_replacement(
            input_nodes,
            csr_row_ptr,
            csr_col_ind,
            csr_weight_ptr,
            max_sample_count,
            None,
        )
        local_sample_offset_start = local_sample_offset_tensor[check_node_idx]
        local_sample_offset_end = local_sample_offset_tensor[check_node_idx + 1]
        local_check_center_localids = local_center_localid[
            local_sample_offset_start:local_sample_offset_end
        ]
        assert local_check_center_localids.unique().size(0) == 1
        return local_sample_output[local_sample_offset_start:local_sample_offset_end]

    expect_hist = gen_hist_with_lambda(max_iter, csr_row_ptr.size(0) - 1, expect_fun)
    actual_hist = gen_hist_with_lambda(max_iter, csr_row_ptr.size(0) - 1, actual_fun)
    check_two_sample_hist(max_iter, expect_hist, actual_hist)
    print("check weighted sample success")


def test_random_node_csr_weighted_sample(
    max_iter, max_sample_count, num_nodes: int, num_edges: int, target_nodes_num: int
):
    create_time = time.time()
    (
        csr_row_ptr,
        csr_col_ind,
        csr_weight,
        target_node_tensor,
    ) = create_random_csr_graph_and_target_nodes(num_nodes, num_edges, target_nodes_num)
    print(
        " create_random_csr_graph_and_target_nodes run time is {} s ".format(
            time.time() - create_time
        )
    )
    check_function_of_weighted_sample(
        max_iter,
        max_sample_count,
        target_node_tensor,
        csr_row_ptr,
        csr_col_ind,
        csr_weight,
    )


if __name__ == "__main__":
    max_sample_count = 30
    neighbor_count = 1000
    max_iter = 3000
    print("test_weighted_sample_without_replacement_with_uniform_weight : ")
    test_single_weighted_sample_without_replacement_with_uniform_weight(
        max_sample_count, neighbor_count, max_iter
    )

    print("test_weighted_sample_without_replacement_with_random_weight : ")
    test_single_weighted_sample_without_replacement_with_random_weight(
        max_sample_count, neighbor_count, max_iter
    )

    graph_num_nodes = 1000
    graph_num_edge = 200000
    target_nodes_num = 512
    test_random_node_csr_weighted_sample(
        max_iter, max_sample_count, graph_num_nodes, graph_num_edge, target_nodes_num
    )
    max_sample_count = 1030
    neighbor_count = 1500
    max_iter = 500
    print("test_weighted_sample_without_replacement_with_uniform_weight LARGE : ")
    test_single_weighted_sample_without_replacement_with_uniform_weight(
        max_sample_count, neighbor_count, max_iter
    )
    print("test_weighted_sample_without_replacement_with_random_weight LARGE : ")
    test_single_weighted_sample_without_replacement_with_random_weight(
        max_sample_count, neighbor_count, max_iter
    )
    graph_num_nodes = 2000
    graph_num_edge = 3000000
    target_nodes_num = 500
    test_random_node_csr_weighted_sample(
        max_iter, max_sample_count, graph_num_nodes, graph_num_edge, target_nodes_num
    )

# step 1: generate inputs: input_nodes,csr_row_ptr,csr_col_ind,csr_weight_ptr
# step 2: generate hist
# step 3: calculate cdf : perform prefix_sum on hist
# step 4: https://docs.scipy.org/doc/scipy/tutorial/stats/continuous_kstwo.html calculate Dn (ks_state) p-values = kstwo.sf(Dn, n)
# if (p-values) >0.95 Accept the hypothesis that the two distributions are identical
#
