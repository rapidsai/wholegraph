/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

namespace whole_graph {

typedef enum {
  OPT_TYPE_SGD = 0,
  OPT_TYPE_LAZY_ADAM = 1,
  OPT_TYPE_RMSPROP = 2,
  OPT_TYPE_ADAGRAD = 3,
} OptimizerType;

struct SGDInfo {
  float weight_decay;// = 0.0f;
};
struct LazyAdamInfo {
  float weight_decay;// = 0.0f;
  float epsilon;     // = 1e-8;
  float beta1;       // = 0.9;
  float beta2;       // = 0.999;
};

struct RMSPropInfo {
  float weight_decay;// = 0.0f;
  float epsilon;     // = 1e-8;
  float alpha;       // = 0.99;
};

struct AdaGradInfo {
  float weight_decay;// = 0.0f;
  float epsilon;     // = 1e-8;
};

struct LazyAdamData {
  float beta1t = 1.0f;
  float beta2t = 1.0f;
};

struct OptimizerInfo {
  OptimizerType type;
  float lr;
  union {
    SGDInfo sgd_info;
    LazyAdamInfo lazy_adam_info;
    RMSPropInfo rms_prop_info;
    AdaGradInfo ada_grad_info;
  } private_info;
};

}// namespace whole_graph