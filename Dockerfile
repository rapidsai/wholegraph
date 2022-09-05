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

FROM nvcr.io/nvidia/pytorch:21.09-py3
RUN pip3 install ogb pyyaml mpi4py
RUN apt-get update && apt install -y gdb pybind11-dev git

RUN FORCE_CUDA=1 pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
RUN git clone --recurse-submodules https://github.com/dmlc/dgl.git -b 0.8.2
RUN cd dgl && \
        mkdir build && \
        cd build && \
        cmake -DUSE_CUDA=ON -DUSE_NCCL=ON -DBUILD_TORCH=ON .. && \
        make -j && \
        cd ../python && \
        python setup.py install
ENV USE_TORCH_ALLOC 1

RUN pip3 install torchmetrics
RUN conda install -y 'cmake>=3.23.1'
