
function(find_and_configure_nvshmem)


CPMAddPackage(NAME nvshmem
                URL "https://developer.download.nvidia.cn/compute/redist/nvshmem/2.9.0/source/nvshmem_src_2.9.0-2.tar.xz")
target_link_libraries(nvshmem  PRIVATE rt pthread dl) 
target_link_libraries(nvshmem_host  PRIVATE rt pthread dl) 
target_link_libraries(nvshmem_device  PRIVATE rt pthread dl) 

endfunction()


find_and_configure_nvshmem()