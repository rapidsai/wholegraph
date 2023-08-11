/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <unistd.h>
#include <getopt.h>
#include <sys/time.h>

#include <iostream>
#include <string>
#include <string_view>

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_op.h>

#include "../common/wholegraph_benchmark.hpp"
#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/env_func_ptrs.hpp"
#include "wholememory/initialize.hpp"

#include "../../tests/wholememory/wholememory_test_utils.hpp"
#include "../../tests/wholememory_ops/embedding_test_utils.hpp"

namespace wholegraph::bench::gather_scatter{

typedef struct GatherScatterBenchParam {
  wholememory_matrix_description_t get_embedding_desc() const
  { 
    int64_t embedding_entry_count = get_embedding_entry_count();
    int64_t matrix_sizes[2] = {embedding_entry_count, embedding_dim};
    return wholememory_create_matrix_desc(
      matrix_sizes, embedding_stride, embedding_storage_offset, embedding_type);
  }
  wholememory_array_description_t get_indices_desc() const
  {
    int64_t indices_count = get_indices_count();
    return wholememory_create_array_desc(indices_count, indices_storage_offset, indices_type);
  }
  wholememory_matrix_description_t get_output_desc() const
  {
    int64_t indices_count = get_indices_count();
    int64_t output_sizes[2] = {indices_count, embedding_dim};
    return wholememory_create_matrix_desc(
      output_sizes, output_stride, output_storage_offset, output_type);
  }

  int64_t get_embedding_granularity() const
  {
    return embedding_stride * wholememory_dtype_get_element_size(embedding_type);
  }

  int64_t get_embedding_table_size() const {
    return embedding_table_size;
  }
  int64_t get_gather_size() const {
    return gather_size;
  }

  wholememory_memory_type_t get_memory_type() const {
    return memory_type;
  }

  wholememory_memory_location_t get_memory_location() const {
    return memory_location;
  }
  int get_loop_count () const {
    return loop_count;
  }
  std::string get_test_type() const {
    return test_type;
  }

  int64_t get_embedding_dim() const {
    return embedding_dim;
  }
  wholememory_dtype_t get_embedding_type() const {
    return embedding_type;
  }

  GatherScatterBenchParam& set_memory_type(wholememory_memory_type_t new_memory_type)
  {
    memory_type = new_memory_type;
    return *this;
  }
  GatherScatterBenchParam& set_memory_location(wholememory_memory_location_t new_memory_location)
  {
    memory_location = new_memory_location;
    return *this;
  }
  GatherScatterBenchParam& set_embedding_table_size(int64_t new_embedding_table_size) {
    int64_t entry_size =  wholememory_dtype_get_element_size(embedding_type) * get_embedding_dim();
    embedding_table_size = (new_embedding_table_size + entry_size - 1) / entry_size * entry_size;
    return *this;
  }
  GatherScatterBenchParam& set_gather_size(int64_t new_gather_size) {
    int64_t entry_size =  wholememory_dtype_get_element_size(embedding_type) * get_embedding_dim();
    gather_size = (new_gather_size + entry_size - 1) / entry_size * entry_size;
    return *this;
  }
  GatherScatterBenchParam& set_embedding_dim(int64_t new_embedding_dim) {
    embedding_dim = new_embedding_dim;
    if (embedding_stride != embedding_dim) embedding_stride = embedding_dim;
    if (output_stride != embedding_dim) output_stride = embedding_dim;
    int64_t entry_size = wholememory_dtype_get_element_size(embedding_type) * embedding_dim;
    embedding_table_size = (embedding_table_size + entry_size - 1) / entry_size * entry_size;
    gather_size = (gather_size + entry_size - 1) / entry_size * entry_size;
    return *this;
  }
  
  GatherScatterBenchParam& set_loop_count(int new_loop_count) {
    loop_count = new_loop_count;
    return *this;
  }
  
  GatherScatterBenchParam& set_test_type(std::string new_test_type) {
    test_type = new_test_type;
    return *this;
  }


  private: 
  int64_t get_embedding_entry_count() const {
    return embedding_table_size / wholememory_dtype_get_element_size(embedding_type) / embedding_dim;
  }
  int64_t get_indices_count() const {
    return gather_size / wholememory_dtype_get_element_size(embedding_type)/ embedding_dim;
  }

  GatherScatterBenchParam& set_embedding_stride(int64_t new_embedding_stride)
  {
    embedding_stride = new_embedding_stride;
    return *this;
  }
  GatherScatterBenchParam& set_output_stride(int64_t new_output_stride)
  {
    output_stride = new_output_stride;
    return *this;
  }
  GatherScatterBenchParam& set_embedding_type(wholememory_dtype_t new_embedding_type)
  {
    embedding_type = new_embedding_type;
    return *this;
  }
  GatherScatterBenchParam& set_indices_type(wholememory_dtype_t new_indices_type)
  {
    indices_type = new_indices_type;
    return *this;
  }
  GatherScatterBenchParam& set_output_type(wholememory_dtype_t new_output_type)
  {
    output_type = new_output_type;
    return *this;
  }
  wholememory_memory_type_t memory_type         = WHOLEMEMORY_MT_CHUNKED;
  wholememory_memory_location_t memory_location = WHOLEMEMORY_ML_DEVICE;
  int64_t embedding_table_size = 1024000LL;
  int64_t gather_size = 1024; 
  int64_t embedding_dim                         = 32;
  int loop_count = 20;
  std::string test_type = "gather"; //gather or scatter

  int64_t embedding_stride                      = 32;
  int64_t output_stride                         = 32;
  wholememory_dtype_t embedding_type            = WHOLEMEMORY_DT_FLOAT;
  wholememory_dtype_t indices_type              = WHOLEMEMORY_DT_INT64;
  wholememory_dtype_t output_type               = WHOLEMEMORY_DT_FLOAT;
  int64_t embedding_storage_offset              = 0;
  int64_t indices_storage_offset                = 0;
  int64_t output_storage_offset                 = 0;
} GatherScatterBenchParam;

std::string get_memory_type_string(wholememory_memory_type_t memory_type) {
  std::string str; 
  switch (memory_type)
  {
  case WHOLEMEMORY_MT_NONE:
    str = "WHOLEMEMORY_MT_NONE";
    break;
  case WHOLEMEMORY_MT_CONTINUOUS:
    str = "WHOLEMEMORY_MT_CONTINUOUS";
    break;
  case WHOLEMEMORY_MT_CHUNKED:
    str = "WHOLEMEMORY_MT_CHUNKED";
    break;
  case WHOLEMEMORY_MT_DISTRIBUTED:
    str = "WHOLEMEMORY_MT_DISTRIBUTED";
    break;
  default:
    break;
  }
  return str;
}

std::string get_memory_location_string(wholememory_memory_location_t memory_location) {
  std::string str;
  switch (memory_location)
  {
  case WHOLEMEMORY_ML_NONE:
    str = "WHOLEMEMORY_ML_NONE";
    break;
  case WHOLEMEMORY_ML_DEVICE:
    str =  "WHOLEMEMORY_ML_DEVICE";
    break;
  case WHOLEMEMORY_ML_HOST:
    str = "WHOLEMEMORY_ML_HOST";
    break;
  default:
    break;
  }
  return str;
}

void gather_scatter_benchmark(GatherScatterBenchParam &params) {
  int g_dev_count = ForkGetDeviceCount(); 
  EXPECT_GE(g_dev_count, 1);
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, g_dev_count);
  MultiProcessRun(
    g_dev_count,
    [&params, &pipes](int world_rank, int world_size) {
      wholememory_init(0);

      cudaSetDevice(world_rank);

      wholememory_comm_t wm_comm = create_communicator_by_pipes(pipes, world_rank, world_size);

      auto embedding_desc = params.get_embedding_desc();
      auto indices_desc = params.get_indices_desc();
      auto output_desc = params.get_output_desc();
      std::string test_type = params.get_test_type();
      size_t embedding_entry_size = params.get_embedding_granularity();
      wholememory_handle_t embedding_handle;
      wholememory_malloc(&embedding_handle,
                                   params.get_embedding_table_size(),
                                   wm_comm,
                                   params.get_memory_type(),
                                   params.get_memory_location(),
                                   embedding_entry_size);

      cudaStream_t stream;
      cudaStreamCreate(&stream);

      void *dev_indices = nullptr, *dev_gather_buffer = nullptr;
      void *host_indices = nullptr;
      size_t gather_buffer_size  = params.get_gather_size();
      size_t indices_buffer_size = wholememory_get_memory_size_from_array(&indices_desc);

      cudaMallocHost(&host_indices, indices_buffer_size);
      cudaMalloc(&dev_indices, indices_buffer_size);
      cudaMalloc(&dev_gather_buffer, gather_buffer_size);

      wholememory_ops::testing::device_random_init_local_embedding_table(
        embedding_handle, embedding_desc, stream);
      wholememory_ops::testing::host_random_init_indices(
        host_indices, indices_desc, embedding_desc.sizes[0]);
      cudaMemcpyAsync(dev_indices,
                                host_indices,
                                wholememory_get_memory_size_from_array(&indices_desc),
                                cudaMemcpyHostToDevice,
                                stream);
      cudaStreamSynchronize(stream);
      wholememory_communicator_barrier(wm_comm);

      wholememory_tensor_t embedding_tensor;
      wholememory_tensor_description_t embedding_tensor_desc;
      wholememory_copy_matrix_desc_to_tensor(&embedding_tensor_desc, &embedding_desc);
      wholememory_make_tensor_from_handle(
                  &embedding_tensor, embedding_handle, &embedding_tensor_desc);

      wholememory_tensor_t indices_tensor, output_tensor;
      wholememory_tensor_description_t indices_tensor_desc, output_tensor_desc;
      wholememory_copy_array_desc_to_tensor(&indices_tensor_desc, &indices_desc);
      wholememory_copy_matrix_desc_to_tensor(&output_tensor_desc, &output_desc);
      wholememory_make_tensor_from_pointer(&indices_tensor, dev_indices, &indices_tensor_desc);
      wholememory_make_tensor_from_pointer(
                  &output_tensor, dev_gather_buffer, &output_tensor_desc);
      cudaStreamSynchronize(stream);
      wholememory_communicator_barrier(wm_comm);

      for (int i = 0; i < 10; i++) {
        if (test_type.compare("gather") == 0) {
          wholememory_gather(embedding_tensor,
                             indices_tensor,
                             output_tensor,
                             wholememory::get_default_env_func(),
                             stream);
        }
        else if (test_type.compare("scatter") == 0) {
          wholememory_scatter(output_tensor,
                              indices_tensor,
                              embedding_tensor,
                              wholememory::get_default_env_func(),
                              stream);
        }
      }
      cudaStreamSynchronize(stream);
      cudaDeviceSynchronize();
      wholememory_communicator_barrier(wm_comm);

      int loop_count = params.get_loop_count();
      struct timeval tv_s, tv_e;
      if (test_type.compare("gather") == 0) {
        gettimeofday(&tv_s, nullptr);
        for (int i = 0; i < loop_count; i++) {
          wholememory_gather(embedding_tensor,
                            indices_tensor,
                            output_tensor,
                            wholememory::get_default_env_func(),
                            stream);
        }
        cudaStreamSynchronize(stream);
        gettimeofday(&tv_e, nullptr);
      }
      else if (test_type.compare("scatter") == 0) {
          gettimeofday(&tv_s, nullptr);
          for (int i = 0; i < loop_count; i++) {
          wholememory_scatter(output_tensor,
                                      indices_tensor,
                                      embedding_tensor,
                                      wholememory::get_default_env_func(),
                                      stream);
          }
          cudaStreamSynchronize(stream);
          gettimeofday(&tv_e, nullptr);
      }
      else {
        throw std::invalid_argument("Invalid test type");
      }
      int time_us = TIME_DIFF_US(tv_s, tv_e);
      double bw = gather_buffer_size / time_us / 1e3 * loop_count;
      wholememory_communicator_barrier(wm_comm);

      std::vector<double> recv_vec(world_size);
      wm_comm->host_allgather(&bw, recv_vec.data(), 1, WHOLEMEMORY_DT_DOUBLE);
      
      double min_bw, max_bw, avg_bw;
      min_bw = max_bw = recv_vec[0];
      avg_bw = 0.0;
      for (int i = 0; i < world_size; i++) {
          min_bw = std::min(min_bw, recv_vec[i]);
          max_bw = std::max(max_bw, recv_vec[i]);
          avg_bw += recv_vec[i];
      }
      avg_bw /= world_size;
      double emb_size_mb = (double)params.get_embedding_table_size()/1024.0/1024.0;
      double gather_size_mb = (double)params.get_gather_size()/1024.0/1024.0;
      if (world_rank == 0) {
        printf("%s, world_size=%d, memoryType=%s, memoryLocation=%s, elt_size=%ld, embeddingDim=%ld, embeddingTableSize=%.2lf MB, gatherSize=%.2lf MB, minBW=%.2lf GB/s, maxBW=%.2lf GB/s, "
            "avgBW=%.2lf GB/s\n",
            test_type.c_str(), world_size, get_memory_type_string(params.get_memory_type()).c_str(), get_memory_location_string(params.get_memory_location()).c_str(), wholememory_dtype_get_element_size(params.get_embedding_type()), params.get_embedding_dim(), emb_size_mb, gather_size_mb, min_bw, max_bw,
            avg_bw);
      }

      wholememory_destroy_tensor(indices_tensor);
      wholememory_destroy_tensor(output_tensor);

      cudaFreeHost(host_indices);
      cudaFree(dev_indices);
      cudaFree(dev_gather_buffer);

      wholememory_destroy_tensor(embedding_tensor);

      wholememory_free(embedding_handle);

      wholememory::destroy_all_communicators();

      wholememory_finalize();
    },
    true);
}

}  // namespace wholegraph::bench::gather_catter

int main(int argc, char** argv) {
    wholegraph::bench::gather_scatter::GatherScatterBenchParam params;
    const char* optstr = "ht:l:e:g:d:c:f:";
    struct option opts[] = {
        {"help", no_argument, NULL, 'h'},
        {"memory_type", required_argument, NULL, 't'},        // 0: None, 1: Continuous, 2: Chunked, 3 Distributed
        {"memory_location", required_argument, NULL, 'l'},    // 0: None, 1: Device, 2: Host
        {"embedding_table_size", required_argument, NULL, 'e'},
        {"gather_size", required_argument, NULL, 'g'},
        {"embedding_dim", required_argument, NULL, 'd'},
        {"loop_count", required_argument, NULL, 'c'},
        {"test_type", required_argument, NULL, 'f'}                   //test_type: gather or scatter
    };

    const char *usage = "Usage: %s [options]\n"
                  "Options:\n"
                  "  -h, --help      display this help and exit\n"
                  "  -t, --memory_type   specify wholememory type, 0: None, 1: Continuous, 2: Chunked, 3: Distributed\n"
                  "  -l, --memory_location    specify wholememory location, 0: None, 1: Device, 2: Host\n"
                  "  -e, --embedding_table_size    specify embedding table size\n"
                  "  -g, --gather_size    specify gather size\n"
                  "  -d, --embedding_dim    specify embedding dimension\n"
                  "  -c, --loop_count    specify loop count\n"
                  "  -f, --test_type    specify test type: gather or scatter\n";

    int c;
    bool has_option = false;
    while((c = getopt_long(argc, argv, optstr, opts, NULL)) != -1) {
        has_option = true;
        switch (c)
        {
        char *endptr;
        long val;
        case 'h':
            printf(usage, argv[0]);
            exit(EXIT_SUCCESS);
        case 't':
            val = strtol(optarg,&endptr, 10);
            if (*endptr != '\0' || val < 0 || val > 3) {
                printf("Invalid argument for option -t\n");
                printf(usage, argv[0]);
                exit(EXIT_FAILURE);
            }
              params.set_memory_type(static_cast<wholememory_memory_type_t>(val));
            break;
        case 'l':
            val = strtol(optarg,&endptr, 10);
            if (*endptr != '\0' || val < 0 || val > 2) {
                printf("Invalid argument for option -l\n");
                printf(usage, argv[0]);
                exit(EXIT_FAILURE);
            }
            params.set_memory_location(static_cast<wholememory_memory_location_t>(val));
            break;
        case 'e':
            try {
                long long val = std::stoll(optarg);
                if (val < 0) {
                    throw std::invalid_argument("Negative value");
                }
                params.set_embedding_table_size(val);
            }
            catch (std::exception& e) {
                printf("Invalid argument for option -e\n");
                printf(usage, argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        case 'g':
            try {
                long long val = std::stoll(optarg);
                if (val < 0) {
                    throw std::invalid_argument("Negative value");
                }
                params.set_gather_size(val);
            }
            catch (std::exception& e) {
                printf("Invalid argument for option -g\n");
                printf(usage, argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        case 'd':
            try {
                int val = std::stoll(optarg);
                if (val < 0) {
                    throw std::invalid_argument("Negative value");
                }
                params.set_embedding_dim(val);
            }
            catch (std::exception& e) {
                printf("Invalid argument for option -d\n");
                printf(usage, argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        case 'c':
            try {
                int val = std::stoi(optarg);
                if (val < 0) {
                    throw std::invalid_argument("Negative value");
                }
                params.set_loop_count(val);
            }
            catch (std::exception& e) {
                printf("Invalid argument for option -c\n");
                printf(usage, argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        case 'f':
            if (strcmp(optarg, "gather") == 0) {
                params.set_test_type("gather");
            }
            else if (strcmp(optarg, "scatter") == 0) {
                params.set_test_type("scatter");
            }
            else {
                printf("Invalid argument for option -f\n");
                printf(usage, argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        default:
            printf("Invalid or unrecognized option\n");
            printf(usage, argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (!has_option) {
      printf("No option or argument is passed, use the default param\n");
    }
    wholegraph::bench::gather_scatter::gather_scatter_benchmark(params);
    return 0;
}