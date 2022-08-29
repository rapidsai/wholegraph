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

from enum import IntEnum
from typing import Union

import torch
from wg_torch.wm_tensor import *

from wholegraph.torch import wholegraph_pytorch as wg


def embedding_2d_sub_tensor_assign(
    t: torch.Tensor, emb: Union[torch.Tensor, wg.ChunkedTensor], start_idx: int
):
    assert t.dim() == 2
    assert emb.dim() == 2
    assert t.shape[1] == emb.shape[1]
    if isinstance(emb, torch.Tensor):
        emb[start_idx : start_idx + t.shape[0]] = t
    else:
        wg.chunked_embedding_2d_sub_tensor_assign(t, emb, start_idx)


class EmbeddingOptimizerTypes(IntEnum):
    OPT_TYPE_SGD = 0
    OPT_TYPE_LAZY_ADAM = 1
    OPT_TYPE_RMSPROP = 2
    OPT_TYPE_ADAGRAD = 3
    OPT_TYPE_NONE = 4


class EmbeddingOptimizer(object):
    def __init__(self):
        super(EmbeddingOptimizer, self).__init__()
        self.opt_type = EmbeddingOptimizerTypes.OPT_TYPE_NONE
        self.opt_data = []

    def get_type(self):
        return self.opt_type

    def get_opt_data(self):
        return self.opt_data


class EmbeddingSGDOptimizer(EmbeddingOptimizer):
    def __init__(self, weight_decay=0.0):
        super(EmbeddingSGDOptimizer, self).__init__()
        self.opt_type = EmbeddingOptimizerTypes.OPT_TYPE_SGD
        self.opt_data = [weight_decay]


class EmbeddingLazyAdamOptimizer(EmbeddingOptimizer):
    def __init__(self, weight_decay=0.0, epsilon=1e-8, beta1=0.9, beta2=0.999):
        super(EmbeddingLazyAdamOptimizer, self).__init__()
        self.opt_type = EmbeddingOptimizerTypes.OPT_TYPE_LAZY_ADAM
        self.opt_data = [weight_decay, epsilon, beta1, beta2]


class EmbeddingRMSPropOptimizer(EmbeddingOptimizer):
    def __init__(self, weight_decay=0.0, epsilon=1e-8, alpha=0.99):
        super(EmbeddingRMSPropOptimizer, self).__init__()
        self.opt_type = EmbeddingOptimizerTypes.OPT_TYPE_RMSPROP
        self.opt_data = [weight_decay, epsilon, alpha]


class EmbeddingAdaGradOptimizer(EmbeddingOptimizer):
    def __init__(self, weight_decay=0.0, epsilon=1e-8):
        super(EmbeddingAdaGradOptimizer, self).__init__()
        self.opt_type = EmbeddingOptimizerTypes.OPT_TYPE_ADAGRAD
        self.opt_data = [weight_decay, epsilon]


def get_sizes_for_optimizer(opt_type: EmbeddingOptimizerTypes):
    # returns count of per_element_states, per_embedding_states and bitmaps
    if opt_type == EmbeddingOptimizerTypes.OPT_TYPE_SGD:
        return 0, (0, 0, None), 1
    elif opt_type == EmbeddingOptimizerTypes.OPT_TYPE_LAZY_ADAM:
        return 2, (1, 2, torch.float32), 1
    elif opt_type == EmbeddingOptimizerTypes.OPT_TYPE_RMSPROP:
        return 1, (0, 0, None), 1
    elif opt_type == EmbeddingOptimizerTypes.OPT_TYPE_ADAGRAD:
        return 1, (0, 0, None), 1
    else:
        raise TypeError("optimizer type can't be none.")


def create_optimizer_states_collective(
    embedding: Union[torch.Tensor, wg.ChunkedTensor, wg.NCCLTensor],
    optimizer_type: EmbeddingOptimizerTypes,
    wm_tensor_type: WmTensorType = WmTensorType.CHUNKED,
    force_dtype: Union[torch.dtype, None] = None,
):
    assert optimizer_type is not EmbeddingOptimizerTypes.OPT_TYPE_NONE
    dtype = embedding.dtype if force_dtype is None else force_dtype
    assert len(embedding.shape) == 2
    entry_count = embedding.shape[0]
    embedding_dim = embedding.shape[1]
    embedding_stride = embedding.stride()[0]
    wm_comm = get_wm_communicator(embedding)

    def create_tensor_fn(sizes, strides=[], tensor_dtype=dtype):
        return create_wm_tensor(wm_comm, sizes, strides, tensor_dtype, wm_tensor_type)

    per_element_count, per_embedding_data, bit_map_count = get_sizes_for_optimizer(
        optimizer_type
    )
    per_embedding_count, per_embedding_elt_count, emb_state_dtype = per_embedding_data

    embedding_sizes = [entry_count, embedding_dim]

    per_element_states = []
    per_embedding_states = []

    for i in range(per_element_count):
        per_element_states.append(
            create_tensor_fn(sizes=embedding_sizes, strides=[embedding_stride, 1])
        )
        lt = get_local_tensor(per_element_states[-1])
        torch.nn.init.constant_(lt, 0.0)
    for i in range(per_embedding_count):
        per_embedding_states.append(
            create_tensor_fn(
                sizes=[entry_count, per_embedding_elt_count], tensor_dtype=torch.float32
            )
        )
        lt = get_local_tensor(per_embedding_states[-1])
        torch.nn.init.constant_(lt, 1.0)
    return per_element_states, per_embedding_states


def apply_embedding_gradients_collective(
    opt_type: EmbeddingOptimizerTypes,
    learning_rate: float,
    optimizer_data: list,
    local_sparse_indice: torch.Tensor,
    local_sparse_grad: torch.Tensor,
    embedding: Union[torch.Tensor, wg.ChunkedTensor, wg.NCCLTensor],
    per_element_states: list,
    per_embedding_states: list,
):
    per_element_count, per_embedding_data, _ = get_sizes_for_optimizer(opt_type)
    per_embedding_count, _, _ = per_embedding_data

    assert len(per_element_states) == per_element_count
    assert len(per_embedding_states) == per_embedding_count

    local_embedding = get_local_tensor(embedding)
    local_per_element_states = [get_local_tensor(t) for t in per_element_states]
    local_per_per_embedding_states = [get_local_tensor(t) for t in per_embedding_states]

    wg.embedding_apply_gradients_collective(
        int(opt_type),
        learning_rate,
        optimizer_data,
        local_sparse_indice,
        local_sparse_grad,
        local_embedding,
        local_per_element_states,
        local_per_per_embedding_states,
    )


embedding_backward_comm = None
trainable_wholememory_embedding_array = None
embedding_optimizer = None


class TrainableEmbedding(object):
    def __init__(
        self,
        embedding: Union[torch.Tensor, wg.ChunkedTensor, wg.NCCLTensor],
        force_dtype: Union[torch.dtype, None] = None,
    ):
        super(TrainableEmbedding, self).__init__()
        self.embedding = embedding
        self.force_dtype = force_dtype
        global embedding_optimizer
        (
            self.per_element_states,
            self.per_embedding_states,
        ) = create_optimizer_states_collective(
            embedding=self.embedding,
            optimizer_type=embedding_optimizer.get_type(),
            force_dtype=self.force_dtype,
            wm_tensor_type=get_wm_tensor_type(self.embedding),
        )
        global trainable_wholememory_embedding_array
        trainable_wholememory_embedding_array.append(self)
        self.need_backward = False
        self.sparse_indices = []
        self.sparse_grads = []

    @property
    def shape(self):
        return self.embedding.shape

    @property
    def dtype(self):
        return self.embedding.dtype

    def apply(self, learning_rate: float):
        sparse_indices = torch.cat(self.sparse_indices)
        sparse_grads = torch.cat(self.sparse_grads)
        bcomm = get_wm_communicator(self.embedding)
        (
            local_sparse_indice,
            local_sparse_grad,
        ) = torch.ops.wholegraph.exchange_embedding_grads(
            sparse_indices, sparse_grads, self.embedding.shape[0], bcomm
        )
        global embedding_optimizer
        apply_embedding_gradients_collective(
            embedding_optimizer.get_type(),
            learning_rate=learning_rate,
            optimizer_data=embedding_optimizer.get_opt_data(),
            local_sparse_indice=local_sparse_indice,
            local_sparse_grad=local_sparse_grad,
            embedding=self.embedding,
            per_element_states=self.per_element_states,
            per_embedding_states=self.per_embedding_states,
        )

        self.sparse_indices = []
        self.sparse_grads = []


def embedding_lookup_nograd_common(
    embedding_table: Union[torch.Tensor, wg.ChunkedTensor, wg.NCCLTensor],
    indice: torch.Tensor,
):
    if isinstance(embedding_table, torch.Tensor):
        out_tensor = torch.ops.wholegraph.gather(
            indice, embedding_table, embedding_table.dtype
        )
    elif isinstance(embedding_table, wg.ChunkedTensor):
        out_tensor = torch.ops.wholegraph.gather_chunked(
            indice, embedding_table.get_ptr(), embedding_table.dtype
        )
    else:
        out_tensor = torch.ops.wholegraph.gather_nccl(indice, embedding_table.get_ptr())
    return out_tensor


def scatter_nograd(
    input_tensor: torch.Tensor,
    indice: torch.Tensor,
    embedding_table: Union[torch.Tensor, wg.ChunkedTensor, wg.NCCLTensor],
):
    if isinstance(embedding_table, torch.Tensor):
        torch.ops.wholegraph.scatter(input_tensor, indice, embedding_table)
    elif isinstance(embedding_table, wg.ChunkedTensor):
        torch.ops.wholegraph.scatter_chunked(
            input_tensor, indice, embedding_table.get_ptr()
        )
    else:
        assert isinstance(embedding_table, wg.NCCLTensor)
        torch.ops.wholegraph.scatter_nccl(
            input_tensor, indice, embedding_table.get_ptr()
        )


class EmbeddingLookupFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indice: torch.Tensor,
        embedding_table: Union[
            torch.Tensor, wg.ChunkedTensor, wg.NCCLTensor, TrainableEmbedding
        ],
        dummy_input: Union[torch.Tensor or None] = None,
        dtype: Union[torch.dtype, None] = None,
    ):
        real_embedding_table = (
            embedding_table.embedding
            if isinstance(embedding_table, TrainableEmbedding)
            else embedding_table
        )
        real_dtype = real_embedding_table.dtype
        out_dtype = real_dtype
        if dtype is not None:
            out_dtype = dtype

        if isinstance(embedding_table, TrainableEmbedding) and dummy_input is not None:
            ctx.save_for_backward(indice, dummy_input)
            ctx.et = embedding_table
            embedding_table.need_backward = True
            if isinstance(real_embedding_table, torch.Tensor):
                out_tensor = torch.ops.wholegraph.gather_need_grad(
                    indice, real_embedding_table, out_dtype
                )
            elif isinstance(real_embedding_table, wg.ChunkedTensor):
                out_tensor = torch.ops.wholegraph.gather_chunked_need_grad(
                    indice, real_embedding_table.get_ptr(), out_dtype
                )
            else:
                assert isinstance(real_embedding_table, wg.NCCLTensor)
                assert out_dtype == real_dtype
                out_tensor = torch.ops.wholegraph.gather_nccl_need_grad(
                    indice, real_embedding_table.get_ptr()
                )
        else:
            if isinstance(real_embedding_table, torch.Tensor):
                out_tensor = torch.ops.wholegraph.gather(
                    indice, real_embedding_table, out_dtype
                )
            elif isinstance(real_embedding_table, wg.ChunkedTensor):
                out_tensor = torch.ops.wholegraph.gather_chunked(
                    indice, real_embedding_table.get_ptr(), out_dtype
                )
            else:
                out_tensor = torch.ops.wholegraph.gather_nccl(
                    indice, real_embedding_table.get_ptr()
                )
                if out_dtype != real_dtype:
                    out_tensor = out_tensor.to(out_dtype)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        if ctx.saved_tensors is None or len(ctx.saved_tensors) == 0:
            return None, None, None
        indice, dummy_input = ctx.saved_tensors
        embedding_table = ctx.et
        dummy_input_grad = torch.zeros_like(dummy_input)
        assert isinstance(embedding_table, TrainableEmbedding)
        ctx.et = None
        assert (
            indice.dim() == 1
            and grad_outputs.dim() == 2
            and indice.shape[0] == grad_outputs.shape[0]
        )
        assert grad_outputs.shape[1] == embedding_table.embedding.shape[1]
        embedding_table.sparse_indices.append(indice)
        embedding_table.sparse_grads.append(grad_outputs)
        return None, None, dummy_input_grad


class EmbeddingLookUpModule(torch.nn.Module):
    def __init__(self, need_backward=True):
        super().__init__()
        self.need_backward = need_backward
        if need_backward:
            self.dummy_weight = torch.nn.Parameter(
                torch.zeros(1), requires_grad=need_backward
            )
        else:
            self.dummy_weight = None
        self.embedding_lookup_fn = EmbeddingLookupFn.apply

    def forward(
        self,
        indice: torch.Tensor,
        embedding_table: Union[torch.Tensor, wg.ChunkedTensor, TrainableEmbedding],
    ):
        if self.need_backward:
            return self.embedding_lookup_fn(indice, embedding_table, self.dummy_weight)
        else:
            with torch.no_grad():
                return self.embedding_lookup_fn(
                    indice, embedding_table, self.dummy_weight
                )


def init_embedding_backward_env(
    barrier_comm: int, optimizer=EmbeddingLazyAdamOptimizer()
):
    global embedding_backward_comm
    assert embedding_backward_comm is None
    embedding_backward_comm = barrier_comm
    global trainable_wholememory_embedding_array
    assert trainable_wholememory_embedding_array is None
    trainable_wholememory_embedding_array = []
    global embedding_optimizer
    assert embedding_optimizer is None
    embedding_optimizer = optimizer


def run_optimizers(lr):
    # reading embeddings and writing gradients
    global trainable_wholememory_embedding_array
    for te in trainable_wholememory_embedding_array:
        if te.need_backward is True:
            te.apply(learning_rate=lr)
            te.need_backward = False
    # update embeddings and reset gradients
    global embedding_backward_comm
    wg.barrier(embedding_backward_comm)
    torch.cuda.synchronize()


def finalize_embedding_backward_env():
    global embedding_backward_comm
    wg.barrier(embedding_backward_comm)
    embedding_backward_comm = None
    global trainable_wholememory_embedding_array
    trainable_wholememory_embedding_array = []
