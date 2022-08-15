add_executable(whole_memory_mp_test whole_memory_mp_test.cu)
target_link_libraries(whole_memory_mp_test whole_graph)
set_property(TARGET whole_memory_mp_test PROPERTY CUDA_ARCHITECTURES 70-real 80-real 86-real 70-virtual)

add_executable(whole_memory_sp_test whole_memory_sp_test.cu)
target_link_libraries(whole_memory_sp_test whole_graph)
set_property(TARGET whole_memory_sp_test PROPERTY CUDA_ARCHITECTURES 70-real 80-real 86-real 70-virtual)

add_executable(whole_memory_perf_scan whole_memory_perf_scan.cu)
target_link_libraries(whole_memory_perf_scan whole_graph)
set_property(TARGET whole_memory_perf_scan PROPERTY CUDA_ARCHITECTURES 70-real 80-real 86-real 70-virtual)

add_executable(bootstrap_test bootstrap_test.cu)
target_link_libraries(bootstrap_test whole_graph)
set_property(TARGET bootstrap_test PROPERTY CUDA_ARCHITECTURES 70-real 80-real 86-real 70-virtual)

add_executable(block_topk_test block_topk_test.cu)
set_property(TARGET block_topk_test PROPERTY CUDA_ARCHITECTURES 70-real 80-real 86-real 70-virtual)
