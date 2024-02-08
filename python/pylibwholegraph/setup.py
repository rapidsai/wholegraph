# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

import os

from setuptools import find_packages, Command
from skbuild import setup


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = [
        ("all", None, None),
    ]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        setupFileDir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(setupFileDir)
        os.system("rm -rf build")
        os.system("rm -rf _skbuild")
        os.system("rm -rf dist")
        os.system('find . -name "__pycache__" -type d -exec rm -rf {} +')
        os.system("rm -rf *.egg-info")
        os.system('find . -name "*.cpython*.so" -type f -delete')


cmdclass = dict()
cmdclass["clean"] = CleanCommand


setup(
    packages=find_packages(
        include=[
            "pylibwholegraph",
            "pylibwholegraph.*",
        ],
        exclude=["*tests*"],
    ),
    package_data={
        "pylibwholegraph": ["VERSION", "torch_cpp_ext/*.cpp",
                            "torch_cpp_ext/*.h"],
    },
    include_package_data=True,
    cmdclass=cmdclass,
    zip_safe=False,
)
