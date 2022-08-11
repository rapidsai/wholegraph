#include "file_utils.h"

#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <string.h>

#include <iostream>
#include <string>

namespace whole_graph {

bool IsFileExist(const std::string& filename, int mode) {
  return access(filename.c_str(), mode) == 0;
}

size_t StatFileSize(const std::string& filename) {
  auto filesize = (size_t)-1;
  struct stat statbuf{};
  if (stat(filename.c_str(), &statbuf) < 0) {
    return filesize;
  } else {
    filesize = statbuf.st_size;
  }
  return filesize;
}

bool IsDirExist(const std::string& dirname) {
  DIR* dp = opendir(dirname.c_str());
  if (dp == NULL) return false;
  closedir(dp);
  return true;
}

bool GetFileListFromDir(const std::string& dirname, std::vector<std::string>* filelist) {
  filelist->clear();
  if (!IsDirExist(dirname)) return false;
  DIR* dp = opendir(dirname.c_str());
  assert(dp != NULL);

  struct dirent* fileindir = NULL;
  while ((fileindir = readdir(dp)) != NULL) {
    if (strcmp(fileindir->d_name, ".") == 0 || strcmp(fileindir->d_name, "..") == 0)
      continue;
    if (fileindir->d_type == DT_REG) {
      filelist->push_back(fileindir->d_name);
    }
  }
  closedir(dp);
  return true;
}

std::string JoinPath(const std::string& path1, const std::string& path2) {
  std::string path = path1;
  if (path.size() == 0 || path[path.size() - 1] != '/') path.append("/");
  path.append(path2);
  return path;
}

void SplitPathAndFile(const std::string& path_and_name, std::string* path, std::string* filename) {
  size_t last = path_and_name.find_last_of('/');
  if (last == std::string::npos) {
    *path = ".";
    *filename = path_and_name;
    return;
  }
  *path = path_and_name.substr(0, last);
  *filename = path_and_name.substr(last + 1);
}

std::string GetPartFileName(const std::string& prefix, int part_id, int part_count) {
  std::string filename = prefix;
  filename.append("_part_").append(std::to_string(part_id)).append("_of_").append(std::to_string(part_count));
  return filename;
}

bool GetPartFileListFromPrefix(const std::string& prefix, std::vector<std::string>* filelist) {
  filelist->clear();
  std::string path, file_prefix;
  SplitPathAndFile(prefix, &path, &file_prefix);
  std::vector<std::string> files_in_dir;
  GetFileListFromDir(path, &files_in_dir);
  std::vector<std::string> part_files;
  for (const auto& file_in_dir : files_in_dir) {
    if (file_in_dir.rfind(file_prefix, 0) == 0) {
      part_files.push_back(file_in_dir);
    }
  }
  int file_count = part_files.size();
  if (file_count == 0) {
    std::cerr << "file_count is 0\n";
    return false;
  }
  if (file_count == 1) {
    if (files_in_dir[0] == file_prefix) {
      filelist->push_back(prefix);
      return true;
    }
  }
  std::vector<int> file_indice(file_count, -1);
  for (int i = 0; i < file_count; i++) {
    std::string filename = GetPartFileName(file_prefix, i, file_count);
    for (size_t j = 0; j < part_files.size(); j++) {
      if (part_files[j] == filename) {
        file_indice[i] = j;
      }
    }
    if (file_indice[i] == -1) {
      std::cerr << "file of indice " << i << " not found.\n";
      return false;
    }
  }
  for (int i = 0; i < file_count; i++) {
    std::string filename = GetPartFileName(prefix, i, file_count);
    filelist->push_back(filename);
  }
  return true;
}

}