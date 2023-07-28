# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

from __future__ import print_function
import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess

import clangutils
from gitutils import modifiedFiles


GPU_ARCH_REGEX = re.compile(r"sm_(\d+)")
SPACES = re.compile(r"\s+")
# depfiles are tricky. We first replace '\[any space char]' with \0 which
# cannot appear in any valid file-name. Then split, and replace \0 with space
DEP_FILE_ESCAPE = re.compile(r"\\\s")
DEP_FILE_REPL = "\0"
DEP_FILE_ESCAPE_ANY = re.compile(r"\\(.)")
XCOMPILER_FLAG = re.compile(r"-((Xcompiler)|(-compiler-options))=?")
XPTXAS_FLAG = re.compile(r"-((Xptxas)|(-ptxas-options))=?")
# any options that may have equal signs in nvcc but not in clang
# add those options here if you find any
OPTIONS_NO_EQUAL_SIGN = ['-isystem']
SEPARATOR = "-" * 8
END_SEPARATOR = "*" * 64
FP_INSTANCE = re.compile(r"_(bf|fp)(8|16|32|64|128)")
I_INSTANCE = re.compile(r"_(s|u)(8|16|32|64|128)")
EIDX_INSTANCE = re.compile(r"_eidx")


def parse_args():
    argparser = argparse.ArgumentParser("Runs clang-tidy on a cmake project")
    argparser.add_argument(
        "cdb", nargs='+',
        help="Path to cmake-generated compilation database(s)")
    argparser.add_argument(
        "-exe", type=str, default="clang-tidy", help="Path to clang-tidy exe")
    argparser.add_argument(
        "-ignore", type=str, default=None,
        help="Regex used to ignore files from checking")
    argparser.add_argument(
        "-select", type=str, default=None,
        help="Regex used to select files for checking")
    argparser.add_argument(
        "-j", type=int, default=-1,
        help="Number of parallel jobs to launch. "
        "If this is <= 0, it is set to CPU core count")
    argparser.add_argument(
        "-root", type=str, default=None,
        help="Root path to cmake build files, which can be separate from "
        "repo root. It must be a common root for all compilation databases. "
        "By default, the working directory of this script "
        "(which must be the git repo root).")
    argparser.add_argument(
        "-git_modified_only", action="store_true",
        help="If set, only check files that were modified in the current PR "
        "(CI environment) or uncommited files (non-CI environment).")
    argparser.add_argument(
        "-check_once", action="store_true",
        help="If set, we attempt to check instantiations at most once. "
        "Useful for local development, since it is much faster and only "
        "warnings in specific template instantiations will not be caught.")
    argparser.add_argument("-launcher", type=str, default=None,
        help="Compiler launcher such as ccache or sccache. By default, none.")
    argparser.add_argument("-header", nargs="*", required=False,
        help="Name(s) - not entire paths - of headers to include. "
        "They must be part of the default sources except a select list "
        "(see clangutils script for details and config).")
    argparser.add_argument("-warn", choices=["none", "all", "same"],
        default="all",
        help="If set to 'none', do not report any compiler warnings. "
        "If set to 'all' (current default), report all compiler warnings. "
        "If set to 'same', report as in command from compilation database.")
    argparser.add_argument(
        "-v", action="store_true", help="Verbose output.")
    args = argparser.parse_args()
    if args.j <= 0:
        args.j = mp.cpu_count()
    args.ignore_compiled = re.compile(args.ignore) if args.ignore else None
    args.select_compiled = re.compile(args.select) if args.select else None
    args.compiler = shutil.which(clangutils.CLANG_COMPILER)
    if args.compiler is None:
        raise Exception(
            "Unable to find clang compiler %s" % clangutils.CLANG_COMPILER
        )
    args.exe = shutil.which(args.exe)
    if args.exe is None:
        raise Exception("Unable to find clang-tidy %s" % args.exe)
    # we check clang's version so that it will work in CI
    clangutils.check_clang_version(args.compiler)
    for cdb in args.cdb:
        if not os.path.isfile(cdb):
            raise Exception("Compilation database '%s' missing" % cdb)
    # by default, CDB root is also the repo root (current working directory)
    if args.root is None:
        args.root = os.getcwd()
    args.root = os.path.realpath(os.path.expanduser(args.root))
    if args.header:
        args.headers = set(args.header)
    else:
        args.headers = set(
            os.path.basename(h) for h in clangutils.list_all_headers()
        )
    # get modified files if necessary
    args.modified_files = dict()
    if args.git_modified_only:
        args.modified_files = {
            os.path.realpath(
                os.path.expanduser(os.path.join(args.root, f))
            ): True
            for f in modifiedFiles()
        }
    elif args.check_once:
        args.modified_files = {
            f: True for f in clangutils.list_all_src_files()
        }

    if args.v:
        print("Using {} ({} processes)".format(
            args.compiler, args.j))
        print("Using git modified files only: {}".format(
            args.git_modified_only))
        print("ROOT dir: {}\nLauncher: {}".format(args.root, args.launcher))
        print("Ignore regex: {}\nSelect regex: {}".format(
            args.ignore_compiled, args.select_compiled))
    return args


def get_no_instance_path(f):
    f_dir, f_name = os.path.split(f)
    f_name_base = FP_INSTANCE.sub("", f_name)
    f_name_base = I_INSTANCE.sub("", f_name_base)
    f_name_base = EIDX_INSTANCE.sub("", f_name_base)
    return os.path.join(f_dir, f_name_base)


def update_include_search_dirs(command, root):
    # first we extract (and remove) paths from CPATH, C_INCLUDE_PATH and
    # CPLUS_INCLUDE_PATH
    env = os.environ.copy()
    dirs = []
    for var in ["CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"]:
        p = env.pop(var, "")
        dirs.extend(d for d in p.split(os.pathsep) if d and os.path.isdir(d))
    to_remove = []
    for i, flag in enumerate(command):
        if flag.startswith("-I") and os.path.isdir(flag[2:]):
            to_remove.append(i)
            dirs.append(flag[2:])
    for i in sorted(to_remove, reverse=True):
        del command[i]

    default_dirs = clangutils.get_default_dirs_with_sources(
        [root, os.getcwd()]
    )
    new_dir_cmds = []
    for d in dirs:
        if any(os.path.commonpath([dd, d]) == dd for dd in default_dirs):
            new_dir_cmds.append("-I" + d)
        else:
            new_dir_cmds.extend(["-isystem", d])

    command[1:1] = new_dir_cmds
    return env


def get_clang_arch_flag(command):
    # clang only accepts a single architecture, and does not distinguish
    # between virtual and physical architecture.
    # So we just list all architecture numbers, then get the minimum value
    archs = []
    for loc in range(len(command)):
        if (command[loc] != "-gencode" and command[loc] != "--generate-code"
                and not command[loc].startswith("--generate-code=")):
            continue
        if command[loc].startswith("--generate-code="):
            arch_flag = command[loc][len("--generate-code="):]
        else:
            arch_flag = command[loc + 1]
        match = GPU_ARCH_REGEX.search(arch_flag)
        if match is not None:
            archs.append(int(match.group(1)))
    return "--cuda-gpu-arch=sm_%d" % min(archs)


def get_index(arr, item_options):
    return set(i for i, s in enumerate(arr) for item in item_options
               if s == item)


def remove_items(arr, item_options):
    for i in sorted(get_index(arr, item_options), reverse=True):
        del arr[i]


def remove_items_plus_one(arr, item_options):
    for i in sorted(get_index(arr, item_options), reverse=True):
        if i < len(arr) - 1:
            del arr[i + 1]
        del arr[i]
    idx = set(i for i, s in enumerate(arr) for item in item_options
              if s.startswith(item + "="))
    for i in sorted(idx, reverse=True):
        del arr[i]


def add_cuda_path(command, nvcc):
    # Check if we are using conda compilers. If yes, we need to use cuda-gdb
    # path as nvcc path refer to the fake nvcc shell script in the conda env
    # Modified in WholeGraph: always use cuda-gdb path.
    # if "_build_env" in nvcc:
    #     nvcc_path = shutil.which("cuda-gdb")
    # else:
    #     nvcc_path = shutil.which(nvcc)
    nvcc_path = shutil.which("cuda-gdb")
    if not nvcc_path:
        raise Exception("Command %s has invalid compiler %s" % (command, nvcc))
    cuda_root = os.path.dirname(os.path.dirname(nvcc_path))
    command.append('--cuda-path=%s' % cuda_root)


def get_tidy_args(cmd, gcc_root, launcher, compiler, warn, root):
    command, f_path = cmd["command"], cmd["file"]
    is_cuda = f_path.endswith(".cu")
    command = SPACES.split(command)
    # get and replace original compiler
    cc_orig = command[0]
    command[0] = compiler
    # either add -Werror, remove -Werror flags or keep command as-is
    if warn == "all":
        # treat all compiler warnings as errors
        # however, we never want to warn about the CUDA version or command line
        # itself to be able to support newer features from CUDA
        command[1:1] = [
            "-Werror",
            "-Wno-error=unknown-cuda-version",
            "-Wno-error=unused-command-line-argument"
        ]
    elif warn == "none":
        # remove any -Werror flags
        for i, x in reversed(list(enumerate(command))):
            if x.startswith("-Werror"):
                del command[i]
    # in any case, move -I to -isystem if the paths are not below our default
    # dirs to avoid reporting warnings from other libraries
    env = update_include_search_dirs(command, root)

    if launcher:
        command.insert(0, launcher)
    # remove compilation and output targets from the original command
    remove_items_plus_one(command, ["--compile", "-c"])
    remove_items_plus_one(command, ["--output-file", "-o"])
    if is_cuda:
        # replace nvcc's "-gencode ..." with clang's "--cuda-gpu-arch ..."
        # also, clang only supports single arch, so we use the lowest one
        command.append(get_clang_arch_flag(command))
        # provide proper cuda path to clang
        add_cuda_path(command, cc_orig)
        # remove all kinds of nvcc flags clang doesn't know about
        remove_items_plus_one(command, [
            "--generate-code",
            "-gencode",
            "--x",
            "-x",
            "--compiler-bindir",
            "-ccbin",
            "--diag_suppress",
            "-diag-suppress",
            "--default-stream",
            "-default-stream",
            "--Werror",
        ])
        remove_items(command, [
            "-extended-lambda",
            "--extended-lambda",
            "-expt-extended-lambda",
            "--expt-extended-lambda",
            "-expt-relaxed-constexpr",
            "--expt-relaxed-constexpr",
            "--device-debug",
            "-G",
            "--generate-line-info",
            "-lineinfo",
        ])
        # "-x cuda" is the right usage in clang
        command.extend(["-x", "cuda"])
        # we remove -Xcompiler flags: here we basically have to hope for the
        # best that clang++ will accept any flags which nvcc passed to gcc
        for i, c in reversed(list(enumerate(command))):
            new_c = XCOMPILER_FLAG.sub('', c)
            if new_c == c:
                continue
            command[i:i + 1] = new_c.split(',')
        # we also change -Xptxas to -Xcuda-ptxas, always adding space here
        for i, c in reversed(list(enumerate(command))):
            if XPTXAS_FLAG.search(c):
                if not c.endswith("=") and i < len(command) - 1:
                    del command[i + 1]
                command[i] = '-Xcuda-ptxas'
                command.insert(i + 1, XPTXAS_FLAG.sub('', c))
        # several options like isystem don't expect `=`
        for opt in OPTIONS_NO_EQUAL_SIGN:
            opt_eq = opt + '='
            # make sure that we iterate from back to front here for insert
            for i, c in reversed(list(enumerate(command))):
                if not c.startswith(opt_eq):
                    continue
                x = c.split('=')
                # we only care about the first `=`
                command[i] = x[0]
                command.insert(i + 1, '='.join(x[1:]))
        # use extensible whole program, to avoid ptx resolution/linking
        command.extend(["-Xcuda-ptxas", "-ewp"])
        # for libcudacxx, we need to allow variadic functions
        command.extend(["-Xclang", "-fcuda-allow-variadic-functions"])
        # add some additional CUDA intrinsics
        cuda_intrinsics_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "__clang_cuda_additional_intrinsics.h")
        command.extend(["-include", cuda_intrinsics_file])
    # remove flags for NVCC/GCC that clang doesn't know about
    remove_items(command, [
        "--forward-unknown-to-host-compiler",
        "-forward-unknown-to-host-compiler",
        "-fvar-tracking-assignments"
    ])
    # try to figure out which GCC CMAKE used, and tell clang all about it
    command.append("--gcc-toolchain=%s" % gcc_root)
    return command, is_cuda, env


def check_output_for_errors(output):
    # there shouldn't really be any allowed errors
    warnings_found = 0
    errors = []
    for line in output.splitlines():
        if line.find("error:") >= 0:
            errors.append(line)
        if line.find("warning:") >= 0:
            warnings_found += 1
    return warnings_found, errors


def run_clang_tidy_command(tidy_cmd, cwd, env):
    cmd = " ".join(tidy_cmd)
    result = subprocess.run(cmd, check=False, shell=True, cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result.stdout = result.stdout.decode("utf-8").strip()
    out = "CMD: " + cmd + "\n"
    out += "EXIT-CODE: %d\n" % result.returncode
    n_warnings, errors = check_output_for_errors(result.stdout)
    status = n_warnings == 0 and not errors
    out += result.stdout
    return status, out, errors


def get_dependencies(clang_cmd, f_path, cwd):
    # -MM prints user dependency files (not system) and stops after
    # pre-processor. We also set the name of main file output, just to make
    # sure it will be included in the dependencies itself
    dep_cmd = " ".join(clang_cmd + ["-MM", "-MT" + f_path, f_path])
    # we cannot capture warnings/errors here since parsing output is difficult
    # if any error happens, we have to rely on return code and will just
    # re-run things
    result = subprocess.run(dep_cmd, check=False, shell=True, cwd=cwd,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        subprocess.check_call(dep_cmd, shell=True, cwd=cwd)
        # make sure that we raise here no matter what
        e = ("Got EXIT-CODE %d while trying to get dependencies of file "
             "%s\nCMD: %s\nCWD: %s")
        raise Exception(e % (result.returncode, f_path, clang_cmd, cwd))
    # first replace all escaped spaces with special character
    result = DEP_FILE_ESCAPE.sub(DEP_FILE_REPL, result.stdout.decode("utf-8"))
    deps = set()
    # simply split on spaces
    for dep_name in result.split():
        # replace special character with regular space again
        dep = dep_name.replace(DEP_FILE_REPL, " ")
        # strip and check if anything is left
        dep = dep.strip()
        if not dep:
            continue
        # remove colon if we have one at the end of the file-name
        # also un-escape any other characters
        dep = dep.rstrip(":")
        dep = DEP_FILE_ESCAPE_ANY.sub(r"\1", dep)
        dep = os.path.realpath(os.path.expanduser(os.path.join(cwd, dep)))
        deps.add(dep)
    return deps


class LockContext(object):
    def __init__(self, lock=None) -> None:
        self._lock = lock

    def __enter__(self):
        if self._lock:
            self._lock.acquire()
        return self

    def __exit__(self, _, __, ___):
        if self._lock:
            self._lock.release()
        return False  # we don't handle exceptions


def print_result(passed, stdout, f_name, errors, verbose):
    if any(errors):
        raise Exception(
            "File %s: got %d errors:\n%s" % (f_name, len(errors), stdout))
    status_str = "PASSED" if passed else "FAILED"
    print("%s File:%s %s %s" % (SEPARATOR, f_name, status_str, SEPARATOR))
    if (verbose or not passed) and stdout:
        print(stdout)
        print("%s\n" % END_SEPARATOR)
    return stdout.splitlines() if stdout else []


def run_clang_tidy(lock, modified_files, args, build_dir, db_cmd):
    f_path = db_cmd["file"]
    gcc_root = clangutils.get_gcc_root(args, build_dir)
    cmd, is_cuda, env = get_tidy_args(
        db_cmd, gcc_root, args.launcher, args.compiler, args.warn, args.root
    )
    rel_path = os.path.relpath(f_path, start=args.root)
    # first check if we should skip this file entirely
    if modified_files:
        deps = get_dependencies(cmd, f_path, build_dir)
        mod = set(f for f, valid in modified_files.items() if valid)
        dep_mod = deps.intersection(mod)
        if not dep_mod:
            print("%s File:%s %s %s" % (
                SEPARATOR, rel_path, "SKIPPED", SEPARATOR)
            )
            return True, []
        # remove the intersection + files with same name except fp32/s32 etc.
        if args.check_once:
            f_path_no_inst = get_no_instance_path(f_path)
            for f in mod:
                if f_path_no_inst == get_no_instance_path(f):
                    dep_mod.add(f)
            modified_files.update({f: False for f in dep_mod})

    sub_dirs = "|".join(clangutils.HEADER_SUB_DIRS)
    sep = re.escape(os.sep)
    sub_dirs = sep + "[^/]+" + sep + "(" + sub_dirs + ")" + sep + ".*" + sep
    cpp_modernize = "--extra-arg=-std=c++17"
    header_filter = "--header-filter='.*%s[^%s]+[.](cuh|h|hpp)$'" % (
        os.path.basename(args.root) + sub_dirs, sep
    )
    filter_headers = [{"name": h} for h in args.headers]
    line_filter = "--line-filter='%s'" % json.dumps(filter_headers)
    tidy_cmd = [args.exe, cpp_modernize, header_filter, line_filter, f_path, "--"]
    tidy_cmd.extend(cmd)
    status = True
    out = ""
    if is_cuda:
        tidy_cmd.append("--cuda-device-only")
        tidy_cmd.append(f_path)
        ret, out1, errors1 = run_clang_tidy_command(tidy_cmd, build_dir, env)
        out += out1
        out += "\n%s\n" % SEPARATOR
        status = status and ret
        tidy_cmd[-2] = "--cuda-host-only"
        ret, out1, errors2 = run_clang_tidy_command(tidy_cmd, build_dir, env)
        status = status and ret
        out += out1
        errors = errors1 + errors2
    else:
        tidy_cmd.append(f_path)
        ret, out1, errors = run_clang_tidy_command(tidy_cmd, build_dir, env)
        status = status and ret
        out += out1
    # we immediately print the result since this is more interactive for user
    with lock:
        lines = print_result(status, out, rel_path, errors, args.v)
        return status, lines


def parse_results(results):
    return all(r[0] for r in results), [s for r in results for s in r[1]]


# mostly used for debugging purposes
def run_sequential(args, build_dir, all_cmds):
    lock = LockContext()
    results = []
    # actual tidy checker
    for cmd in all_cmds:
        results.append(
            run_clang_tidy(lock, args.modified_files, args, build_dir, cmd)
        )
    return parse_results(results)


def run_parallel(args, build_dir, all_cmds):
    results = []
    with mp.Manager() as manager:
        lock = manager.Lock()
        if args.check_once:
            modified_files = manager.dict(args.modified_files)
        else:
            modified_files = args.modified_files
        with manager.Pool(args.j) as pool:
            for cmd in all_cmds:
                results.append(
                    pool.apply_async(
                        run_clang_tidy, args=(
                            lock, modified_files, args, build_dir, cmd
                        )
                    )
                )
            results_final = [r.get() for r in results]
    return parse_results(results_final)


def list_all_cmds(args, cdb):
    with open(cdb, "r") as fp:
        all_cmds = json.load(fp)

    to_remove = []
    # ensure that we use only the real paths, filter and get the clang commands
    for i, cmd in enumerate(all_cmds):
        cmd["file"] = os.path.realpath(os.path.expanduser(cmd["file"]))
        if os.path.commonpath([args.root, cmd["file"]]) != args.root:
            # this may happen with dependencies that we build into our
            # libraries/executables like nanobind
            if args.v:
                print(
                    "%s File:%s ignored (not in root %s) %s" % (
                    SEPARATOR, cmd["file"], args.root, SEPARATOR)
                )
            to_remove.append(i)
        if args.ignore_compiled is not None and \
           re.search(args.ignore_compiled, cmd["file"]) is not None:
            to_remove.append(i)
        if args.select_compiled is not None and \
           re.search(args.select_compiled, cmd["file"]) is None:
            to_remove.append(i)

    for i in sorted(to_remove, reverse=True):
        del all_cmds[i]

    return all_cmds


def main():
    args = parse_args()
    if args.git_modified_only and not args.modified_files:
        print("No modified files detected. Nothing to do.")
        return

    # Attempt to making sure that we run this script from root of repo always
    if not os.path.exists(".git"):
        raise Exception("This needs to always be run from the root of repo")
    for cdb in args.cdb:
        build_dir = os.path.dirname(cdb)
        all_cmds = list_all_cmds(args, cdb)
        print("Checking %d files/compilation commands" % len(all_cmds))
        if args.j == 1:
            status, lines = run_sequential(args, build_dir, all_cmds)
        else:
            status, lines = run_parallel(args, build_dir, all_cmds)
        if not status:
            # first get a list of all checks that were run
            ret = subprocess.check_output(
                args.exe + " --list-checks", shell=True
            )
            ret = ret.decode("utf-8")
            checks = [line.strip() for line in ret.splitlines()
                    if line.startswith(' ' * 4)]
            max_check_len = max(len(c) for c in checks)
            check_counts = dict()
            content = os.linesep.join(lines)
            for check in checks:
                check_counts[check] = content.count(check)
            sorted_counts = sorted(
                check_counts.items(), key=lambda x: x[1], reverse=True)
            print("Failed {} check(s) in total. Counts as per below:".format(
                sum(1 for _, count in sorted_counts if count > 0)))
            for check, count in sorted_counts:
                if count <= 0:
                    break
                n_space = max_check_len - len(check) + 4
                print("{}:{}{}".format(check, ' ' * n_space, count))
            raise Exception("clang-tidy failed! Refer to the errors above.")


if __name__ == "__main__":
    main()
