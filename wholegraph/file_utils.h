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

#include <string>
#include <vector>

namespace whole_graph {

bool IsFileExist(const std::string &filename, int mode);
size_t StatFileSize(const std::string &filename);
bool IsDirExist(const std::string &dirname);
bool GetFileListFromDir(const std::string &dirname, std::vector<std::string> *filelist);
std::string JoinPath(const std::string &path1, const std::string &path2);
std::string GetPartFileName(const std::string &prefix, int part_id, int part_count);
bool GetPartFileListFromPrefix(const std::string &prefix, std::vector<std::string> *filelist);

}// namespace whole_graph