#pragma once

#include <string>
#include <vector>

namespace whole_memory {

bool IsFileExist(const std::string& filename, int mode);
size_t StatFileSize(const std::string& filename);
bool IsDirExist(const std::string& dirname);
bool GetFileListFromDir(const std::string& dirname, std::vector<std::string>* filelist);
std::string JoinPath(const std::string& path1, const std::string& path2);
bool GetPartFileListFromPrefix(const std::string& prefix, std::vector<std::string>* filelist);

}