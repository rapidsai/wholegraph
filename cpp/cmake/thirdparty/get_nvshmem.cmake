
function(find_and_configure_nvshmem)

set(NVSHMEM_IBGDA_SUPPORT ON)
set(NVSHMEM_IBDEVX_SUPPORT ON)
set(MLX5_lib /usr/lib/x86_64-linux-gnu/libmlx5.so.1)

# CPMAddPackage(NAME nvshmem
#                 URL "https://developer.download.nvidia.cn/compute/redist/nvshmem/2.9.0/source/nvshmem_src_2.9.0-2.tar.xz")

CPMAddPackage(NAME nvshmem
              URL ${CMAKE_SOURCE_DIR}/../archives/nvshmem_src_2.10.1-3.txz)
target_link_libraries(nvshmem  PRIVATE rt pthread dl) 
target_link_libraries(nvshmem_host  PRIVATE rt pthread dl) 
target_link_libraries(nvshmem_device  PRIVATE rt pthread dl) 

endfunction()


find_and_configure_nvshmem()