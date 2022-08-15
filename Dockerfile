FROM nvcr.io/nvidia/pytorch:21.09-py3
RUN pip3 install ogb pyyaml mpi4py
RUN apt-get update && apt install -y gdb pybind11-dev

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
