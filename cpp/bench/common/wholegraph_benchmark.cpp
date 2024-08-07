/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include "wholegraph_benchmark.hpp"

#include "wholememory/communicator.hpp"
#include <cstdint>
#include <experimental/functional>
#include <experimental/random>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include <functional>
#include <string>
#include <vector>

namespace wholegraph::bench {

template <typename IndexT>
void host_get_random_integer_indices(void* indices,
                                     wholememory_array_description_t indice_desc,
                                     int64_t max_indices)
{
  IndexT* indices_ptr = static_cast<IndexT*>(indices);
  std::experimental::reseed();
  for (int64_t i = 0; i < indice_desc.size; i++) {
    IndexT random_index = std::experimental::randint<IndexT>(0, max_indices - 1);
    indices_ptr[i + indice_desc.storage_offset] = random_index;
  }
}

void host_random_init_integer_indices(void* indices,
                                      wholememory_array_description_t indices_desc,
                                      int64_t max_indices)
{
  if (indices_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_random_integer_indices<int>(indices, indices_desc, max_indices);
  } else {
    host_get_random_integer_indices<int64_t>(indices, indices_desc, max_indices);
  }
}

void host_random_partition(size_t* partition_sizes, size_t total_size, int partition_count)
{
  std::default_random_engine random_engine(0);
  std::uniform_int_distribution<size_t> uniform(90, 100);
  size_t acc_size   = 0;
  size_t random_sum = 0;
  for (int i = 0; i < partition_count; i++) {
    partition_sizes[i] = (size_t)uniform(random_engine);
    random_sum += partition_sizes[i];
  }
  for (int i = 0; i < partition_count; i++) {
    partition_sizes[i] = (size_t)((partition_sizes[i] / (double)random_sum) * total_size);
    acc_size += partition_sizes[i];
  }
  partition_sizes[0] += total_size - acc_size;
}

void MultiProcessMeasurePerformance(std::function<void()> run_fn,
                                    wholememory_comm_t& wm_comm,
                                    const PerformanceMeter& meter,
                                    const std::function<void()>& barrier_fn)
{
  barrier_fn();
  // warm up
  struct timeval tv_warmup_s;
  gettimeofday(&tv_warmup_s, nullptr);
  int64_t target_warmup_time = 1000LL * 1000LL * meter.warmup_seconds;
  while (true) {
    struct timeval tv_warmup_c;
    gettimeofday(&tv_warmup_c, nullptr);
    int64_t time_warmup = TIME_DIFF_US(tv_warmup_s, tv_warmup_c);
    if (time_warmup >= target_warmup_time) break;
    run_fn();
    WHOLEMEMORY_CHECK_NOTHROW(cudaDeviceSynchronize() == cudaSuccess);
  }
  WHOLEMEMORY_CHECK_NOTHROW(cudaDeviceSynchronize() == cudaSuccess);
  barrier_fn();

  // run
  struct timeval tv_run_s, tv_run_e;
  int64_t max_run_us = 1000LL * 1000LL * meter.max_run_seconds;
  gettimeofday(&tv_run_s, nullptr);
  int real_run_count = 0;
  for (int i = 0; i < meter.run_count; i++) {
    run_fn();
    real_run_count++;
    struct timeval tv_run_c;
    gettimeofday(&tv_run_c, nullptr);
    int64_t time_run_used = TIME_DIFF_US(tv_run_s, tv_run_c);
    if (time_run_used >= max_run_us || real_run_count >= meter.run_count) break;
    if (meter.sync) { WHOLEMEMORY_CHECK_NOTHROW(cudaDeviceSynchronize() == cudaSuccess); }
  }
  WHOLEMEMORY_CHECK_NOTHROW(cudaDeviceSynchronize() == cudaSuccess);
  gettimeofday(&tv_run_e, nullptr);
  int64_t real_time_used_us = TIME_DIFF_US(tv_run_s, tv_run_e);
  double single_run_time_us = real_time_used_us;
  single_run_time_us /= real_run_count;
  barrier_fn();

  for (size_t i = 0; i < meter.metrics_.size(); i++) {
    double metric_value = meter.metrics_[i].value;
    if (meter.metrics_[i].invert) {
      metric_value *= single_run_time_us;
      metric_value /= 1e6;
    } else {
      metric_value /= single_run_time_us;
      metric_value *= 1e6;
    }

    std::vector<double> recv_vec(wm_comm->world_size);
    wm_comm->host_allgather(&metric_value, recv_vec.data(), 1, WHOLEMEMORY_DT_DOUBLE);
    double min_metric, max_metric, avg_metric;
    min_metric = max_metric = recv_vec[0];
    avg_metric              = 0.0;
    for (int j = 0; j < wm_comm->world_size; j++) {
      min_metric = std::min(min_metric, recv_vec[j]);
      max_metric = std::max(max_metric, recv_vec[j]);
      avg_metric += recv_vec[j];
    }
    avg_metric /= wm_comm->world_size;
    if (wm_comm->world_rank == 0) {
      fprintf(stderr,
              "== Metric: %20s:  min=%.2lf %s,, max=%.2lf %s,, avg=%.2lf %s\n",
              meter.metrics_[i].name.c_str(),
              min_metric,
              meter.metrics_[i].unit.c_str(),
              max_metric,
              meter.metrics_[i].unit.c_str(),
              avg_metric,
              meter.metrics_[i].unit.c_str());
    }
  }
}

}  // namespace wholegraph::bench
