

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

endfunction()


find_and_configure_nvshmem()
