FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y lsb-core software-properties-common wget libspdlog-dev

#RUN remove old cmake to update
RUN conda remove --force -y cmake
RUN rm -rf /usr/local/bin/cmake && rm -rf /usr/local/lib/cmake && rm -rf /usr/lib/cmake

RUN apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc && \
    export LSB_CODENAME=$(lsb_release -cs) && \
    apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ ${LSB_CODENAME} main" && \
    apt update && apt install -y cmake

# update py for pytest
RUN pip3 install -U py
RUN pip3 install Cython setuputils3 scikit-build nanobind pytest-forked pytest
