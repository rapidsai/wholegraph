/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cstdarg>

#include <iostream>
#include <string>

#include <cassert>
#include <raft/core/error.hpp>

#include "error.hpp"
#include <wholememory/wholememory.h>

namespace wholememory {

LogLevel& get_log_level();

void set_log_level(LogLevel lev);

bool will_log_for(LogLevel lev);

/**
 * @defgroup CStringFormat Expand a C-style format string
 *
 * @brief Expands C-style formatted string into std::string
 *
 * @param[in] fmt format string
 * @param[in] vl  respective values for each of format modifiers in the string
 *
 * @return the expanded `std::string`
 *
 * @{
 */
inline std::string format(const char* fmt, va_list& vl)
{
  va_list vl_copy;
  va_copy(vl_copy, vl);
  int length = std::vsnprintf(nullptr, 0, fmt, vl_copy);
  assert(length >= 0);
  std::vector<char> buf(length + 1);
  (void)std::vsnprintf(buf.data(), length + 1, fmt, vl);
  return std::string(buf.data());
}

inline std::string format(const char* fmt, ...)
{
  va_list vl;
  va_start(vl, fmt);
  std::string str = wholememory::format(fmt, vl);
  va_end(vl);
  return str;
}
/** @} */

#define WHOLEMEMORY_LOG(lev, fmt, ...)                                                 \
  do {                                                                                 \
    if (wholememory::will_log_for(lev))                                                \
      std::cout << wholememory::format(fmt, ##__VA_ARGS__) << std::endl << std::flush; \
  } while (0)

#define WHOLEMEMORY_FATAL(fmt, ...)                                                    \
  do {                                                                                 \
    std::string fatal_msg{};                                                           \
    SET_WHOLEMEMORY_ERROR_MSG(fatal_msg, "WholeMemory FATAL at ", fmt, ##__VA_ARGS__); \
    throw wholememory::logic_error(fatal_msg);                                         \
  } while (0)

#define WHOLEMEMORY_ERROR(fmt, ...) WHOLEMEMORY_LOG(LEVEL_ERROR, fmt, ##__VA_ARGS__)
#define WHOLEMEMORY_WARN(fmt, ...)  WHOLEMEMORY_LOG(LEVEL_WARN, fmt, ##__VA_ARGS__)
#define WHOLEMEMORY_INFO(fmt, ...)  WHOLEMEMORY_LOG(LEVEL_INFO, fmt, ##__VA_ARGS__)
#define WHOLEMEMORY_DEBUG(fmt, ...) WHOLEMEMORY_LOG(LEVEL_DEBUG, fmt, ##__VA_ARGS__)
#define WHOLEMEMORY_TRACE(fmt, ...) WHOLEMEMORY_LOG(LEVEL_TRACE, fmt, ##__VA_ARGS__)

}  // namespace wholememory
