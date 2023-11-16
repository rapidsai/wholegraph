#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
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
#=============================================================================


set(USE_NVSHMEM_VERSION 2.10.1)
set(USE_NVSHMEM_VERSION_BRANCH 3)
function(find_and_configure_nvshmem)


    set(NVSHMEM_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})


    rapids_cpm_find(nvshmem ${USE_NVSHMEM_VERSION}
                    GLOBAL_TARGETS nvshmem::nvshmem nvshmem::nvshmem_device nvshmem::nvshmem_host
                    BUILD_EXPORT_SET wholegraph-exports
                    INSTALL_EXPORT_SET wholegraph-exports
                    EXCLUDE_FROM_ALL TRUE
                    CPM_ARGS
                        URL https://developer.download.nvidia.cn/compute/redist/nvshmem/${USE_NVSHMEM_VERSION}/source/nvshmem_src_${USE_NVSHMEM_VERSION}-${USE_NVSHMEM_VERSION_BRANCH}.txz
                        OPTIONS
                            "NVSHMEM_IBGDA_SUPPORT ON"
                            "NVSHMEM_IBDEVX_SUPPORT ON"
                            "NVSHMEM_BUILD_EXAMPLES OFF"
                            "NVSHMEM_BUILD_TESTS OFF"
                            "NVSHMEM_PREFIX  ${NVSHMEM_INSTALL_DIR}"
                    )


    if(NOT TARGET nvshmem::nvshmem AND TARGET nvshmem)
        add_library( nvshmem::nvshmem ALIAS nvshmem)
        add_library(nvshmem::nvshmem_device ALIAS nvshmem_device)
        add_library(nvshmem::nvshmem_host ALIAS nvshmem_host)
    endif()


endfunction()


find_and_configure_nvshmem()
