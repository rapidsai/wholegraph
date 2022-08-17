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

pybind11_add_module(wholegraph_pytorch
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/pytorch_cuda_env_fns.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/pytorch_dtype.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/whole_chunked_pytorch_tensor.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/whole_nccl_pytorch_tensor.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/gather_gpu.cu
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/graph_sampler_gpu.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/gnn_ops_gpu.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/whole_graph_pytorch_wrapper.cc)
target_include_directories(wholegraph_pytorch SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(wholegraph_pytorch PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(wholegraph_pytorch PRIVATE
        pybind11::module
        CUDA::cuda_driver
        CUDA::cudart
        NCCL::NCCL
        ${TORCH_LINKER_LIBS}
        ${TORCH_LIBRARIES}
        ${PYTHON_LIBRARIES}
        whole_graph)

