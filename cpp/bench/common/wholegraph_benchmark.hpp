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
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <sys/time.h>

#include <functional>
#include <string>
#include <vector>

#include "error.hpp"

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>
namespace wholegraph::bench {

#define TIME_DIFF_US(TVS, TVE) \
  ((TVE.tv_sec - TVS.tv_sec) * 1000ULL * 1000ULL + (TVE.tv_usec - TVS.tv_usec))

void host_random_init_integer_indices(void* indices,
                                      wholememory_array_description_t indices_desc,
                                      int64_t max_indices);

void host_random_partition(size_t* partition_sizes, size_t total_size, int partition_count);

struct Metric {
  Metric(const std::string& metrics_name,
         const std::string& metrics_unit,
         const double metrics_value,
         bool inv)
  {
    name   = metrics_name;
    unit   = metrics_unit;
    value  = metrics_value;
    invert = inv;
  }
  std::string name;
  std::string unit;
  double value;
  bool invert;
};

struct PerformanceMeter {
  PerformanceMeter& SetSync()
  {
    sync = true;
    return *this;
  }
  bool sync = false;

  PerformanceMeter& SetWarmupTime(float w)
  {
    warmup_seconds = w;
    return *this;
  }
  float warmup_seconds = 0.05f;

  std::vector<Metric> metrics_;

  PerformanceMeter& AddMetrics(const std::string& metrics_name,
                               const std::string& unit,
                               double value,
                               bool inv = false)
  {
    metrics_.emplace_back(metrics_name, unit, value, inv);
    return *this;
  }

  PerformanceMeter& SetRunCount(int count)
  {
    run_count = count;
    return *this;
  }
  int run_count = 100;

  PerformanceMeter& SetMaxRunSeconds(float sec)
  {
    max_run_seconds = sec;
    return *this;
  }
  float max_run_seconds = 10;

  PerformanceMeter& SetName(const std::string& n)
  {
    name = n;
    return *this;
  }
  std::string name;
};

void MultiProcessMeasurePerformance(std::function<void()> run_fn,
                                    wholememory_comm_t& wm_comm,
                                    const PerformanceMeter& meter,
                                    const std::function<void()>& barrier_fn);

}  // namespace wholegraph::bench
