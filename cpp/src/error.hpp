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
#pragma once

#include <cstdlib>

#include <raft/core/error.hpp>

namespace wholememory {

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * WHOLEMEMORY_EXPECTS and WHOLEMEMORY_FAIL macros.
 *
 */
struct logic_error : public raft::exception {
  explicit logic_error(char const* const message) : raft::exception(message) {}
  explicit logic_error(std::string const& message) : raft::exception(message) {}
};

}  // namespace wholememory

/**
 * Macro to append error message to first argument.
 * This should only be called in contexts where it is OK to throw exceptions!
 */
#define SET_WHOLEMEMORY_ERROR_MSG(msg, location_prefix, fmt, ...)                                \
  do {                                                                                           \
    int const size1 = std::snprintf(nullptr, 0, "%s", location_prefix);                          \
    int const size2 = std::snprintf(nullptr, 0, "file=%s line=%d: ", __FILE__, __LINE__);        \
    int const size3 = std::snprintf(nullptr, 0, fmt, ##__VA_ARGS__);                             \
    if (size1 < 0 || size2 < 0 || size3 < 0) {                                                   \
      (void)printf("Error in snprintf, cannot handle raft exception.\n");                        \
      (void)fflush(stdout);                                                                      \
      abort();                                                                                   \
    }                                                                                            \
    auto size = size1 + size2 + size3 + 1; /* +1 for final '\0' */                               \
    std::vector<char> buf(size);                                                                 \
    (void)std::snprintf(buf.data(), size1 + 1 /* +1 for '\0' */, "%s", location_prefix);         \
    (void)std::snprintf(                                                                         \
      buf.data() + size1, size2 + 1 /* +1 for '\0' */, "file=%s line=%d: ", __FILE__, __LINE__); \
    (void)std::snprintf(                                                                         \
      buf.data() + size1 + size2, size3 + 1 /* +1 for '\0' */, fmt, ##__VA_ARGS__);              \
    msg += std::string(buf.data(), buf.data() + size - 1); /* -1 to remove final '\0' */         \
  } while (0)

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when a condition is false
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] fmt String literal description of the reason that cond is expected to be true with
 * optinal format tags
 * @throw wholememory::logic_error if the condition evaluates to false.
 */
#define WHOLEMEMORY_EXPECTS(cond, fmt, ...)                                                \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      std::string error_msg{};                                                             \
      SET_WHOLEMEMORY_ERROR_MSG(error_msg, "WholeMemory failure at ", fmt, ##__VA_ARGS__); \
      throw wholememory::logic_error(error_msg);                                           \
    }                                                                                      \
  } while (0)

/**
 * @brief Macro for checking (pre-)conditions that abort when a condition is false
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] fmt String literal description of the reason that cond is expected to be true with
 * optinal format tags
 */
#define WHOLEMEMORY_EXPECTS_NOTHROW(cond, fmt, ...)                                        \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      std::string error_msg{};                                                             \
      SET_WHOLEMEMORY_ERROR_MSG(error_msg, "WholeMemory failure at ", fmt, ##__VA_ARGS__); \
      (void)printf("%s\n", error_msg.c_str());                                             \
      (void)fflush(stdout);                                                                \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * @param[in] fmt String literal description of the reason that this code path is erroneous with
 * optinal format tags
 * @throw always throws wholememory::logic_error
 */
#define WHOLEMEMORY_FAIL(fmt, ...)                                                       \
  do {                                                                                   \
    std::string error_msg{};                                                             \
    SET_WHOLEMEMORY_ERROR_MSG(error_msg, "WholeMemory failure at ", fmt, ##__VA_ARGS__); \
    throw wholememory::logic_error(error_msg);                                           \
  } while (0)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * @param[in] fmt String literal description of the reason that this code path is erroneous with
 * optinal format tags, this macro will not throw exceptions but abort the process.
 */
#define WHOLEMEMORY_FAIL_NOTHROW(fmt, ...)                                               \
  do {                                                                                   \
    std::string error_msg{};                                                             \
    SET_WHOLEMEMORY_ERROR_MSG(error_msg, "WholeMemory failure at ", fmt, ##__VA_ARGS__); \
    (void)printf("%s\n", error_msg.c_str());                                             \
    (void)fflush(stdout);                                                                \
    abort();                                                                             \
  } while (0)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * @param[in] X boolean expression to check
 * @throw always throws wholememory::logic_error
 */
#define WHOLEMEMORY_CHECK(X)                                                                      \
  do {                                                                                            \
    if (!(X)) { WHOLEMEMORY_FAIL("File %s, line %d, %s check failed.", __FILE__, __LINE__, #X); } \
  } while (0)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * @param[in] X boolean expression to check
 */
#define WHOLEMEMORY_CHECK_NOTHROW(X)                                                          \
  do {                                                                                        \
    if (!(X)) {                                                                               \
      WHOLEMEMORY_FAIL_NOTHROW("File %s, line %d, %s check failed.", __FILE__, __LINE__, #X); \
    }                                                                                         \
  } while (0)
