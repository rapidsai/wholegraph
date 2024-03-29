# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

import argparse
import datetime
import os
import re
import sys

import git

from fileutils import modifiedFiles

FilesToCheck = [
    re.compile(r"[.](cmake|cpp|cu|cuh|h|hpp|sh|pxd|py|pyx)$"),
    re.compile(r"CMakeLists[.]txt$"),
    re.compile(r"setup[.]cfg$"),
    re.compile(r"meta[.]yaml$"),
]
ExemptFiles = [
    re.compile(r"versioneer[.]py"),
    re.compile(r".*[.]json$"),
    re.compile(r"src/io/gzstream[.]hpp$"),
]

# this will break starting at year 10000, which is probably OK :)
CheckSimple = re.compile(
    r"Copyright *(?:\(c\))? *(\d{4}),? *NVIDIA C(?:ORPORATION|orporation)"
)
CheckDouble = re.compile(
    r"Copyright *(?:\(c\))? *(\d{4})-(\d{4}),? *NVIDIA C(?:ORPORATION|orporation)"  # noqa: E501
)


def checkThisFile(f):
    if isinstance(f, git.Diff):
        if f.deleted_file or f.b_blob.size == 0:
            return False
        f = f.b_path
    elif not os.path.exists(f) or os.stat(f).st_size == 0:
        # This check covers things like symlinks which point to files that DNE
        return False
    for exempt in ExemptFiles:
        if exempt.search(f):
            return False
    for checker in FilesToCheck:
        if checker.search(f):
            return True
    return False


def getCopyrightYears(line):
    res = CheckSimple.search(line)
    if res:
        return int(res.group(1)), int(res.group(1))
    res = CheckDouble.search(line)
    if res:
        return int(res.group(1)), int(res.group(2))
    return None, None


def replaceCurrentYear(line, start, end):
    # first turn a simple regex into double (if applicable). then update years
    res = CheckSimple.sub(r"Copyright (c) \1-\1, NVIDIA CORPORATION", line)
    res = CheckDouble.sub(
        rf"Copyright (c) {start:04d}-{end:04d}, NVIDIA CORPORATION",
        res,
    )
    return res


def checkCopyright(f, update_current_year):
    """Checks for copyright headers and their years."""
    errs = []
    thisYear = datetime.datetime.now().year
    lineNum = 0
    crFound = False
    yearMatched = False

    if isinstance(f, git.Diff):
        path = f.b_path
        lines = f.b_blob.data_stream.read().decode().splitlines(keepends=True)
    else:
        path = f
        with open(f, encoding="utf-8") as fp:
            lines = fp.readlines()

    for line in lines:
        lineNum += 1
        start, end = getCopyrightYears(line)
        if start is None:
            continue
        crFound = True
        if start > end:
            e = [
                path,
                lineNum,
                "First year after second year in the copyright "
                "header (manual fix required)",
                None,
            ]
            errs.append(e)
        elif thisYear < start or thisYear > end:
            e = [
                path,
                lineNum,
                f"Current year {thisYear} not included in the copyright header {start}-{end}",
                None,
            ]
            if thisYear < start:
                e[-1] = replaceCurrentYear(line, thisYear, end)
            if thisYear > end:
                e[-1] = replaceCurrentYear(line, start, thisYear)
            errs.append(e)
        else:
            yearMatched = True
    # copyright header itself not found
    if not crFound:
        e = [
            path,
            0,
            "Copyright header missing or formatted incorrectly "
            "(manual fix required)",
            None,
        ]
        errs.append(e)
    # even if the year matches a copyright header, make the check pass
    if yearMatched:
        errs = []

    if update_current_year:
        errs_update = [x for x in errs if x[-1] is not None]
        if len(errs_update) > 0:
            lines_changed = ", ".join(str(x[1]) for x in errs_update)
            print(f"File: {path}. Changing line(s) {lines_changed}")
            for _, lineNum, __, replacement in errs_update:
                lines[lineNum - 1] = replacement
            with open(path, "w", encoding="utf-8") as out_file:
                out_file.writelines(lines)

    return errs


def getAllFilesUnderDir(root, pathFilter=None):
    retList = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            filePath = os.path.join(dirpath, fn)
            if pathFilter(filePath):
                retList.append(filePath)
    return retList


def checkCopyright_main():
    """
    Checks for copyright headers in all the modified files. In case of local
    repo, this script will just look for uncommitted files and in case of CI
    it compares between branches "$PR_TARGET_BRANCH" and "current-pr-branch"
    """
    retVal = 0

    argparser = argparse.ArgumentParser(
        "Checks for a consistent copyright header in git's modified files"
    )
    argparser.add_argument(
        "--update-current-year",
        dest="update_current_year",
        action="store_true",
        required=False,
        help="If set, "
        "update the current year if a header is already "
        "present and well formatted.",
    )
    argparser.add_argument(
        "--git-modified-only",
        dest="git_modified_only",
        action="store_true",
        required=False,
        help="If set, " "only files seen as modified by git will be " "processed.",
    )

    args, dirs = argparser.parse_known_args()

    if args.git_modified_only:
        files = [f for f in modifiedFiles() if checkThisFile(f)]
    else:
        files = []
        for d in [os.path.abspath(d) for d in dirs]:
            if not os.path.isdir(d):
                raise ValueError(f"{d} is not a directory.")
            files += getAllFilesUnderDir(d, pathFilter=checkThisFile)

    errors = []
    for f in files:
        errors += checkCopyright(f, args.update_current_year)

    if len(errors) > 0:
        if any(e[-1] is None for e in errors):
            print("Copyright headers incomplete in some of the files!")
        for e in errors:
            print("  %s:%d Issue: %s" % (e[0], e[1], e[2]))
        print("")
        n_fixable = sum(1 for e in errors if e[-1] is not None)
        file_from_repo = os.path.relpath(os.path.abspath(__file__))
        if n_fixable > 0 and not args.update_current_year:
            print(
                f"You can run `python {file_from_repo} --git-modified-only "
                "--update-current-year` and stage the results in git to "
                f"fix {n_fixable} of these errors.\n"
            )
        retVal = 1

    return retVal


if __name__ == "__main__":
    sys.exit(checkCopyright_main())
