# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import re
import os
import subprocess

DEFAULT_DIRS = ["cpp"]
ALWAYS_IGNORED_DIRS = ["build", "_skbuild", "_cython_build", "cmake-build-debug"]
HEADER_SUB_DIRS = ["cpp/include", "cpp/src"]
EXCLUDED_HEADER_NAMES = set(["dlpack.h"])
HEADER_EXT = ["h", "hpp", "cuh"]
SRC_EXT_RE = r"[.](cu|cuh|h|hpp|cpp)$"

CLANG_COMPILER = "clang++"
EXPECTED_VERSIONS = ("16.0.0",)
CLANG_VERSION_RE = re.compile(r"(Ubuntu |Debian )?clang version ([0-9.]+)(-[0-9]+)?(~ubuntu[0-9.]+)?")
CLANG_FMT_VERSION_RE = re.compile(r"(Ubuntu |Debian )?clang-format version ([0-9.]+)(-[0-9]+)?(~ubuntu[0-9.]+)?")

GNU_DEFAULT_COMPILER = "g++"
CMAKE_COMPILER_REGEX = re.compile(
    r"^\s*CMAKE_CXX_COMPILER:FILEPATH=(.+)\s*$", re.MULTILINE)


def check_clang_version(compiler_name):
    ret = subprocess.check_output(
        "%s --version" % compiler_name, shell=True
    )
    ret = ret.decode("utf-8")
    version = CLANG_VERSION_RE.match(ret)
    if version is None:
        raise Exception("Failed to figure out clang compiler version!")
    version = version.group(2)
    if version not in EXPECTED_VERSIONS:
        raise Exception("clang compiler version must be in %s found '%s'" %
                        (EXPECTED_VERSIONS, version))


def check_clang_format_version(exe_name):
    ret = subprocess.check_output("%s --version" % exe_name, shell=True)
    ret = ret.decode("utf-8")
    version = CLANG_FMT_VERSION_RE.match(ret)
    if version is None:
        raise Exception("Failed to figure out clang-format version!")
    version = version.group(2)
    if version not in EXPECTED_VERSIONS:
        raise Exception("clang-format version must be in %s found '%s'" %
                        (EXPECTED_VERSIONS, version))


def get_default_dirs_with_sources(roots=None):
    if roots is None:
        roots = ["."]
    dirs = set()
    for r in roots:
        rr = os.path.realpath(os.path.expanduser(r))
        for d in DEFAULT_DIRS:
            for h in HEADER_SUB_DIRS:
                dd = os.path.join(rr, d, h)
                if os.path.isdir(dd):
                    dirs.add(dd)
    return dirs


def list_all_src_files(srcdirs=None, file_re=None, ignore_re=None):
    all_files = []
    if srcdirs is None:
        # we always assume that these scripts are run from repo root
        srcdirs = [
            os.path.realpath(os.path.expanduser(d)) for d in DEFAULT_DIRS
        ]
    if file_re is None:
        file_re = re.compile(SRC_EXT_RE)
    for srcdir in srcdirs:
        for root, dirs, files in os.walk(srcdir):
            if (any(d in root.split(os.sep) for d in ALWAYS_IGNORED_DIRS)):
                continue
            for f in files:
                if re.search(file_re, f):
                    src = os.path.join(root, f)
                    if ignore_re is not None and re.search(ignore_re, src):
                        continue
                    all_files.append(src)
    return all_files


def list_all_headers(srcdirs=None):
    header_re = re.compile("[.]({})$".format("|".join(HEADER_EXT)))
    h_excl = "|".join(re.escape(h) for h in EXCLUDED_HEADER_NAMES)
    excl_re = re.compile("{}({})$".format(re.escape(os.sep), h_excl))
    return list_all_src_files(
        srcdirs=srcdirs, file_re=header_re, ignore_re=excl_re)


def get_gcc_root(args, build_dir):
    # first try to determine GCC based on CMakeCache
    cmake_cache = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.isfile(cmake_cache):
        with open(cmake_cache) as f:
            content = f.read()
        match = CMAKE_COMPILER_REGEX.search(content)
        if match:
            return os.path.dirname(os.path.dirname(match.group(1)))
    # fall-back to g++ install. Note that this might fail on OSes other than
    # Linux, but our build assumes a Linux OS anyway (such as in CI)
    default_gxx = shutil.which(GNU_DEFAULT_COMPILER)
    if default_gxx:
        return os.path.dirname(os.path.dirname(default_gxx))
    raise Exception("Cannot find any g++ install on the system.")
