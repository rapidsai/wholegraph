# Copyright (c) 2024, NVIDIA CORPORATION.

import pylibwholegraph


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(pylibwholegraph.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(pylibwholegraph.__version__, str)
    assert len(pylibwholegraph.__version__) > 0
