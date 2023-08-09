# Copyright (c) 2023, NVIDIA CORPORATION.
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

import git

DEFAULT_DIRS = ["cpp", "pylibwholegraph"]
HEADER_SUB_DIRS = ["benchmarks", "include", "src", "tests", "cpp"]
ALWAYS_IGNORED_DIRS = ["build", "_skbuild"]
HEADER_EXT = ["h", "hpp", "cuh"]
SRC_EXT_RE = r"[.](cu|cuh|h|hpp|cpp)$"


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


def modifiedFiles():
    """Get a set of all modified files, as Diff objects.
    The files returned have been modified in git since the merge base of HEAD
    and the upstream of the target branch. We return the Diff objects so that
    we can read only the staged changes.
    """
    repo = git.Repo()
    # Use the environment variable TARGET_BRANCH or RAPIDS_BASE_BRANCH
    # (defined in CI) if possible
    target_branch = os.environ.get(
        "TARGET_BRANCH", os.environ.get("RAPIDS_BASE_BRANCH")
    )
    if target_branch is None:
        # Fall back to the closest branch if not on CI
        target_branch = repo.git.describe(
            all=True, tags=True, match=["branch-*", "main"], abbrev=0
        ).lstrip("heads/")

    upstream_target_branch = None
    if target_branch in repo.heads:
        # Use the tracking branch of the local reference if it exists. This
        # returns None if no tracking branch is set.
        upstream_target_branch = repo.heads[target_branch].tracking_branch()
    if upstream_target_branch is None:
        # Fall back to the remote with the newest target_branch. This code
        # path is used on CI because the only local branch reference is
        # current-pr-branch, and thus target_branch is not in repo.heads.
        # This also happens if no tracking branch is defined for the local
        # target_branch. We use the remote with the latest commit if
        # multiple remotes are defined.
        candidate_branches = [
            remote.refs[target_branch]
            for remote in repo.remotes
            if target_branch in remote.refs
        ]
        if len(candidate_branches) > 0:
            upstream_target_branch = sorted(
                candidate_branches,
                key=lambda branch: branch.commit.committed_datetime,
            )[-1]
        else:
            # If no remotes are defined, try to use the local version of the
            # target_branch. If this fails, the repo configuration must be very
            # strange and we can fix this script on a case-by-case basis.
            upstream_target_branch = repo.heads[target_branch]
    merge_base = repo.merge_base("HEAD", upstream_target_branch.commit)[0]
    diff = merge_base.diff()
    changed_files = {f for f in diff if f.b_path is not None}
    return changed_files
