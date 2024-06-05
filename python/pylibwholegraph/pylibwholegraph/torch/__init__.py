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

from .comm import (
    WholeMemoryCommunicator,
    create_group_communicator,
    destroy_communicator,
)
from .comm import (
    get_global_communicator,
    get_local_node_communicator,
    get_local_device_communicator,
    split_communicator,
    get_local_mnnvl_communicator,
)

from .embedding import (
    WholeMemoryOptimizer,
    create_wholememory_optimizer,
    destroy_wholememory_optimizer,
)
from .embedding import (
    WholeMemoryCachePolicy,
    create_builtin_cache_policy,
    create_wholememory_cache_policy,
    destroy_wholememory_cache_policy,
)
from .embedding import (
    WholeMemoryEmbedding,
    create_embedding,
    create_embedding_from_filelist,
    destroy_embedding,
)
from .embedding import WholeMemoryEmbeddingModule

from .initialize import init, init_torch_env, init_torch_env_and_create_wm_comm, finalize

from .tensor import (
    WholeMemoryTensor,
    create_wholememory_tensor,
    create_wholememory_tensor_from_filelist,
    destroy_wholememory_tensor,
)
from .graph_structure import GraphStructure

from .utils import get_part_file_name, get_part_file_list

from .distributed_launch import add_distributed_launch_options, distributed_launch
from .distributed_launch import get_rank, get_world_size, get_local_rank, get_local_size

from .common_options import (
    add_common_graph_options,
    add_common_model_options,
    add_common_sampler_options,
)
from .common_options import (
    add_training_options,
    add_dataloader_options,
    add_node_classfication_options,
)

from .gnn_model import set_framework, create_gnn_layers, create_sub_graph, HomoGNNModel
from .data_loader import (
    create_node_claffication_datasets,
    get_train_dataloader,
    get_valid_test_dataloader,
)
from .wholegraph_env import compile_cpp_extension
