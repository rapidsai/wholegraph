# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

import pylibwholegraph.binding.wholememory_binding as wmb
import torch
from .utils import torch_dtype_to_wholememory_dtype, get_file_size
from .utils import str_to_wmb_wholememory_location, str_to_wmb_wholememory_memory_type
from .utils import (
    str_to_wmb_wholememory_optimizer_type,
    str_to_wmb_wholememory_access_type,
)
from typing import Union, List
from .comm import WholeMemoryCommunicator
from .comm import (
    get_global_communicator,
    get_local_node_communicator,
    get_local_device_communicator,
)
from .tensor import WholeMemoryTensor
from .wholegraph_env import wrap_torch_tensor, get_wholegraph_env_fns, get_stream


class WholeMemoryOptimizer(object):
    """
    Sparse Optimizer for WholeMemoryEmbedding.
    Many WholeMemoryEmbedding can share same WholeMemoryOptimizer
    You should not create WholeMemoryOptimizer object directly, but use :func:`create_wholememory_optimizer` instead.
    """

    def __init__(self, global_comm: WholeMemoryCommunicator):
        super().__init__()
        self.wmb_opt = wmb.WholeMemoryOptimizer()
        self.embeddings = []
        self.global_comm = global_comm

    def add_embedding(self, wm_embedding):
        """Add WholeMemory Embedding to this optimizer
        NOTE: you don't need to call this method, it is automatic called when WholeMemory Optimizer is created.
        :param wm_embedding: WholeMemory Embedding that use this optimizer
        :return: None
        """
        assert isinstance(wm_embedding, WholeMemoryEmbedding)
        if wm_embedding.wmb_optimizer is not None:
            raise ValueError("optimizer can only be set once.")
        wm_embedding.wmb_optimizer = self.wmb_opt
        wm_embedding.dummy_input.requires_grad_(True)
        self.wmb_opt.add_embedding(wm_embedding.wmb_embedding)
        self.embeddings.append(wm_embedding)

    def step(self, lr: float):
        r"""Apply gradients to all WholeMemory Embedding that use this optimizer.
        :param lr: learing rate.
        """
        for wm_embedding in self.embeddings:
            if wm_embedding.need_apply:
                wm_embedding.apply_gradients(lr)
        self.global_comm.barrier()


class WholeMemoryCachePolicy(object):
    """
    Cache policy to create WholeMemoryEmbedding.
    NOTE: You should not create WholeMemoryCachePolicy object directly,
    use :func:`create_wholememory_cache_policy` instead.
    """

    def __init__(self, wmb_cache_policy: wmb.WholeMemoryCachePolicy):
        super().__init__()
        self.wmb_cache_policy = wmb_cache_policy


def create_wholememory_cache_policy(
    cache_comm: WholeMemoryCommunicator,
    *,
    memory_type: str = "chunked",
    memory_location: str = "cuda",
    access_type: str = "readonly",
    ratio: float = 0.5,
):
    """
    Create WholeMemoryCachePolicy
    NOTE: in most cases, :func:`create_builtin_cache_policy` can support. This function is a more flexible interface
    :param cache_comm: WholeMemory communicator of the cache
    :param memory_type: WholeMemory type of cache
    :param memory_location: WholeMemory location of cache
    :param access_type: Access type needed
    :param ratio: Ratio of cache
    :return: WholeMemoryCachePolicy
    """
    wmb_cache_policy = wmb.WholeMemoryCachePolicy()
    wmb_cache_policy.create_policy(
        cache_comm.wmb_comm,
        str_to_wmb_wholememory_memory_type(memory_type),
        str_to_wmb_wholememory_location(memory_location),
        str_to_wmb_wholememory_access_type(access_type),
        ratio,
    )
    return WholeMemoryCachePolicy(wmb_cache_policy)


def destroy_wholememory_cache_policy(cache_policy: WholeMemoryCachePolicy):
    """
    Destroy WholeMemoryCachePolicy
    :param cache_policy: WholeMemoryCachePolicy to destroy
    :return: None
    """
    wmb_cache_policy = cache_policy.wmb_cache_policy
    wmb_cache_policy.destroy_policy()
    cache_policy.wmb_cache_policy = None


def create_builtin_cache_policy(
    builtin_cache_type: str,
    embedding_memory_type: str,
    embedding_memory_location: str,
    access_type: str,
    cache_ratio: float,
    *,
    cache_memory_type: str = "",
    cache_memory_location: str = "",
):
    r"""Create builtin cache policy

    :param builtin_cache_type: supported types are none, local_device, local_node and all_devices
    :param embedding_memory_type: WholeMemory type of raw embedding
    :param embedding_memory_location: WholeMemory location of raw embedding
    :param access_type: Access type needed
    :param cache_ratio: Ratio of cache
    :param cache_memory_type: WholeMemory type of cache
    :param cache_memory_location: WholeMemory location of cache
    :return: WholeMemoryCachePolicy or None
    """

    if (
        embedding_memory_type != "continuous"
        and embedding_memory_type != "chunked"
        and embedding_memory_type != "distributed"
        and embedding_memory_type != "hierarchy"
    ):
        raise ValueError(f"embedding_memory_type={embedding_memory_type} is not valid")

    if embedding_memory_location != "cpu" and embedding_memory_location != "cuda":
        raise ValueError(
            f"embedding_memory_location={embedding_memory_location} is not valid"
        )

    if builtin_cache_type == "none":
        return None

    if (
        cache_memory_location != ""
        and cache_memory_location != "cpu"
        and cache_memory_location != "cuda"
    ):
        raise ValueError(
            f"cache_memory_location is {cache_memory_location}, should be empty or cpu, cuda"
        )
    cache_memory_location = (
        "cuda" if cache_memory_location == "" else cache_memory_location
    )
    if builtin_cache_type == "all_devices":
        if embedding_memory_location == "cuda":
            print(
                "[WARNING] Seems you are using device cache for device memory, "
                "this may consume more memory and have low performance than use none cache"
            )
        cache_memory_type = (
            embedding_memory_type if cache_memory_type == "" else cache_memory_type
        )
        return create_wholememory_cache_policy(
            get_global_communicator(),
            memory_type=cache_memory_type,
            memory_location=cache_memory_location,
            access_type=access_type,
            ratio=cache_ratio,
        )

    if builtin_cache_type == "local_node":
        cache_memory_type = "chunked" if cache_memory_type == "" else cache_memory_type
        return create_wholememory_cache_policy(
            get_local_node_communicator(),
            memory_type=cache_memory_type,
            memory_location=cache_memory_location,
            access_type=access_type,
            ratio=cache_ratio,
        )

    if builtin_cache_type == "local_device":
        cache_memory_type = "continuous"
        return create_wholememory_cache_policy(
            get_local_device_communicator(),
            memory_type=cache_memory_type,
            memory_location=cache_memory_location,
            access_type=access_type,
            ratio=cache_ratio,
        )

    raise ValueError(
        f"builtin_cache_type={builtin_cache_type} not supported, "
        f"should be none, local_device, local_node or all_devices"
    )


class EmbeddingLookupFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indice: torch.Tensor,
        dummy_input: torch.Tensor,
        wm_embedding,
        is_training: bool = False,
        force_dtype: Union[torch.dtype, None] = None,
    ):
        output_tensor = wm_embedding.gather(
            indice, is_training=is_training, force_dtype=force_dtype
        )
        if is_training and wm_embedding.need_grad():
            ctx.save_for_backward(indice, output_tensor, dummy_input)
            ctx.wm_embedding = wm_embedding
        return output_tensor

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        indice, output_tensor, dummy_input = ctx.saved_tensors
        wm_embedding = ctx.wm_embedding
        wm_embedding.add_gradients(indice, grad_outputs)
        ctx.wm_embedding = None
        return None, torch.zeros_like(dummy_input), None, None, None


class WholeMemoryEmbedding(object):
    r"""WholeMemory Embedding"""

    def __init__(
        self,
        wmb_embedding: wmb.PyWholeMemoryEmbedding,
        wmb_cache_policy: Union[WholeMemoryCachePolicy, None],
    ):
        super().__init__()
        self.wmb_embedding = wmb_embedding
        self.embedding_tensor = None
        self.optimizer_states = dict()

        self.wmb_cache_policy = wmb_cache_policy

        self.adjust_cache = True if self.wmb_cache_policy is not None else False

        self.wmb_optimizer = None

        self.dummy_input = torch.nn.Parameter(
            torch.zeros(1), requires_grad=False
        )
        self.need_apply = False
        self.sparse_indices = []
        self.sparse_grads = []

    def dim(self):
        return self.get_embedding_tensor().dim()

    @property
    def shape(self):
        return self.get_embedding_tensor().shape

    def set_adjust_cache(self, adjust_cache: bool):
        self.adjust_cache = adjust_cache if self.wmb_cache_policy is not None else False

    def need_grad(self):
        return self.wmb_embedding is not None

    def gather(
        self,
        indice: torch.Tensor,
        *,
        is_training: bool = False,
        force_dtype: Union[torch.dtype, None] = None,
    ):
        assert indice.dim() == 1
        embedding_dim = self.get_embedding_tensor().shape[1]
        embedding_count = indice.shape[0]
        current_cuda_device = "cuda:%d" % (torch.cuda.current_device(),)
        output_dtype = (
            force_dtype if force_dtype is not None else self.embedding_tensor.dtype
        )
        need_grad = self.need_grad() and is_training
        output_tensor = torch.empty(
            [embedding_count, embedding_dim],
            device=current_cuda_device,
            dtype=output_dtype,
            requires_grad=need_grad,
        )
        if need_grad:
            self.need_apply = True
        wmb.EmbeddingGatherForward(
            self.wmb_embedding,
            wrap_torch_tensor(indice),
            wrap_torch_tensor(output_tensor),
            self.adjust_cache,
            get_wholegraph_env_fns(),
            get_stream(),
        )
        return output_tensor

    def add_gradients(self, indice: torch.Tensor, grad_outputs: torch.Tensor):
        # print(f'adding gradients sparse_indices={indice}, sparse_grads={grad_outputs}')
        self.sparse_indices.append(indice)
        self.sparse_grads.append(grad_outputs)

    def apply_gradients(self, lr: float):
        sparse_indices = torch.cat(self.sparse_indices)
        sparse_grads = torch.cat(self.sparse_grads)
        # print(f'applying gradients sparse_indices={sparse_indices}, sparse_grads={sparse_grads}')
        wmb.EmbeddingGatherGradientApply(
            self.wmb_embedding,
            wrap_torch_tensor(sparse_indices),
            wrap_torch_tensor(sparse_grads),
            self.adjust_cache,
            lr,
            get_wholegraph_env_fns(),
            get_stream(),
        )
        self.sparse_indices = []
        self.sparse_grads = []
        self.need_apply = []

    def writeback_all_cache(self):
        self.wmb_embedding.writeback_all_cache(get_stream(False))

    def drop_all_cache(self):
        self.wmb_embedding.drop_all_cache(get_stream(False))

    def get_embedding_tensor(self):
        if self.embedding_tensor is None:
            self.embedding_tensor = WholeMemoryTensor(
                self.wmb_embedding.get_embedding_tensor()
            )
        return self.embedding_tensor

    def get_optimizer_state_names(self):
        return self.wmb_embedding.get_optimizer_state_names()

    def get_optimizer_state(self, state_name):
        if state_name not in self.optimizer_states:
            self.optimizer_states[state_name] = WholeMemoryTensor(
                self.wmb_embedding.get_optimizer_state(state_name)
            )
        return self.optimizer_states[state_name]

    def save(self, file_prefix: str):
        self.get_embedding_tensor().to_file_prefix(file_prefix + "_embedding_tensor")
        for state_name in self.get_optimizer_state_names():
            state = self.get_optimizer_state(state_name)
            state.to_file_prefix(file_prefix + "_" + state_name)

    def load(
        self,
        file_prefix: str,
        *,
        ignore_embedding: bool = False,
        part_count: Union[int, None] = None,
    ):
        if ignore_embedding is False:
            self.get_embedding_tensor().from_file_prefix(
                file_prefix + "_embedding_tensor", part_count
            )
        for state_name in self.get_optimizer_state_names():
            state = self.get_optimizer_state(state_name)
            state.from_file_prefix(file_prefix + "_" + state_name, part_count)


def create_embedding(
    comm: WholeMemoryCommunicator,
    memory_type: str,
    memory_location: str,
    dtype: torch.dtype,
    sizes: List[int],
    *,
    cache_policy: Union[WholeMemoryCachePolicy, None] = None,
    embedding_entry_partition: Union[List[int], None] = None,
    random_init: bool = False,
    gather_sms: int = -1,
    round_robin_size: int = 0
):
    r"""
    Create embedding
    :param comm: WholeMemoryCommunicator
    :param memory_type: WholeMemory type, should be continuous, chunked or distributed
    :param memory_location: WholeMemory location, should be cpu or cuda
    :param dtype: data type
    :param sizes: size of the embedding, must be 2D
    :param cache_policy: cache policy
    :param embedding_entry_partition: rank partition based on entry; embedding_entry_partition[i] determines the
    entry count of rank i and shoud be a positive integer; the sum of embedding_entry_partition should equal to
    total entry count; entries will be equally partitioned if None
    :param gather_sms: the number of SMs used in gather process
    :param round_robin_size: continuous embedding size of a rank using round robin shard strategy
    :return: WholeMemoryEmbedding
    """
    if cache_policy is None:
        wmb_cache_policy = wmb.create_non_cache_policy()
    else:
        wmb_cache_policy = cache_policy.wmb_cache_policy
    assert len(sizes) == 2
    tensor_desc = wmb.PyWholeMemoryTensorDescription()
    tensor_desc.set_dtype(torch_dtype_to_wholememory_dtype(dtype))
    tensor_desc.set_shape(sizes)
    tensor_desc.set_stride([sizes[1], 1])
    if memory_type == 'distributed':
        comm_backend = comm.distributed_backend
        if comm_backend == 'nvshmem' and cache_policy is not None:
            raise AssertionError
        ("The caching feature is not supported yet when using NVSHMEM."
         "Please consider disable it by passing cache_policy = None.")
    if embedding_entry_partition is not None and cache_policy is not None:
        print("embedding_entry_partition is ignored because cache_policy is specified")
        embedding_entry_partition = None
    if embedding_entry_partition is not None and round_robin_size != 0:
        print("round_robin_size is ignored because embedding_entry_partition is specified")
        round_robin_size = 0
    if memory_type == 'hierarchy':  # todo: modified
        comm_backend = comm.distributed_backend
        if comm_backend == 'nvshmem':
            raise AssertionError
        ("Hierarchy embedding is not supported yet when using NVSHMEM.")
        if cache_policy is not None:
            raise AssertionError
        ("Hierarchy embedding is not supported yet when using cache.")
        comm_backend = 'nccl'

    wm_embedding = WholeMemoryEmbedding(
        wmb.create_embedding(
            tensor_desc,
            comm.wmb_comm,
            str_to_wmb_wholememory_memory_type(memory_type),
            str_to_wmb_wholememory_location(memory_location),
            wmb_cache_policy,
            embedding_entry_partition=embedding_entry_partition,
            user_defined_sms=gather_sms,
            round_robin_size=round_robin_size
        ),
        cache_policy,
    )
    if random_init is True:
        (
            local_tensor,
            local_offset,
        ) = wm_embedding.get_embedding_tensor().get_local_tensor()
        torch.nn.init.xavier_uniform_(local_tensor)
    comm.barrier()
    return wm_embedding


def create_embedding_from_filelist(
    comm: WholeMemoryCommunicator,
    memory_type: str,
    memory_location: str,
    filelist: Union[List[str], str],
    dtype: torch.dtype,
    last_dim_size: int,
    *,
    cache_policy: Union[WholeMemoryCachePolicy, None] = None,
    embedding_entry_partition: Union[List[int], None] = None,
    gather_sms: int = -1,
    round_robin_size: int = 0
):
    r"""
    Create embedding from file list
    :param comm: WholeMemoryCommunicator
    :param memory_type: WholeMemory type, should be continuous, chunked or distributed
    :param memory_location: WholeMemory location, should be cpu or cuda
    :param filelist: list of files
    :param dtype: data type
    :param last_dim_size: size of last dim
    :param cache_policy: cache policy
    :param embedding_entry_partition: rank partition based on entry; embedding_entry_partition[i] determines the
    entry count of rank i and shoud be a positive integer; the sum of embedding_entry_partition should equal to
    total entry count; entries will be equally partitioned if None
    :param gather_sms: the number of SMs used in gather process
    :param round_robin_size: continuous embedding size of a rank using round robin shard strategy
    :return:
    """
    if isinstance(filelist, str):
        filelist = [filelist]
    assert last_dim_size > 0
    if embedding_entry_partition is not None and cache_policy is not None:
        print("embedding_entry_partition is ignored because cache_policy is specified")
        embedding_entry_partition = None
    if embedding_entry_partition is not None and round_robin_size != 0:
        print("round_robin_size is ignored because embedding_entry_partition is specified")
        round_robin_size = 0
    element_size = torch.tensor([], dtype=dtype).element_size()
    file_entry_size = element_size * last_dim_size
    total_file_size = 0
    for filename in filelist:
        file_size = get_file_size(filename)
        if file_size % file_entry_size != 0:
            raise ValueError(
                "File %s size is %d not mutlple of %d"
                % (filename, file_size, file_entry_size)
            )
        total_file_size += file_size
    total_entry_count = total_file_size // file_entry_size
    wm_embedding = create_embedding(
        comm,
        memory_type,
        memory_location,
        dtype,
        [total_entry_count, last_dim_size],
        cache_policy=cache_policy,
        embedding_entry_partition=embedding_entry_partition,
        gather_sms=gather_sms,
        round_robin_size=round_robin_size
    )
    wm_embedding.get_embedding_tensor().from_filelist(filelist, round_robin_size)
    return wm_embedding


def destroy_embedding(wm_embedding: WholeMemoryEmbedding):
    """
    Destroy WholeMemoryEmbedding
    :param wm_embedding: WholeMemoryEmbedding to destroy
    :return: None
    """
    wm_embedding.wmb_embedding.destroy_embedding()
    wm_embedding.wmb_embedding = None


class WholeMemoryEmbeddingModule(torch.nn.Module):
    """
    torch.nn.Module wrapper of WholeMemoryEmbedding
    """
    def __init__(self, wm_embedding: WholeMemoryEmbedding):
        super().__init__()
        self.wm_embedding = wm_embedding
        self.embedding_gather_fn = EmbeddingLookupFn.apply

    def forward(
        self, indice: torch.Tensor, force_dtype: Union[torch.dtype, None] = None
    ):
        return self.embedding_gather_fn(
            indice,
            self.wm_embedding.dummy_input,
            self.wm_embedding,
            self.training,
            force_dtype,
        )


def create_wholememory_optimizer(embeddings: Union[WholeMemoryEmbedding, List[WholeMemoryEmbedding]],
                                 optimizer_type: str,
                                 param_dict: dict):
    """
    Create WholeMemoryOptimizer.
    :param embeddings: WholememoryEmbeddings to set the Optimizer
    :param optimizer_type: Type of the Optimizer
    :param param_dict: parameters of the optimizer
    :return: WholeMemoryOptimizer
    """
    wm_optimizer = WholeMemoryOptimizer(get_global_communicator())
    wm_optimizer.wmb_opt.create_optimizer(
        str_to_wmb_wholememory_optimizer_type(optimizer_type), param_dict
    )
    if isinstance(embeddings, WholeMemoryEmbedding):
        wm_optimizer.add_embedding(embeddings)
    else:
        for em in embeddings:
            wm_optimizer.add_embedding(em)
    return wm_optimizer


def destroy_wholememory_optimizer(optimizer: WholeMemoryOptimizer):
    """
    Destroy WholeMemoryOptimizer
    :param optimizer: WholeMemoryOptimizer to destroy
    :return: None
    """
    optimizer.wmb_opt.destroy_optimizer()
    optimizer.wmb_opt = None
