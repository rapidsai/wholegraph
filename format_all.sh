#! /bin/bash

find ./wholegraph -iname "*.h" -o -iname "*.cc" -o -iname "*.cuh" -o -iname "*.cu" | xargs clang-format-10 -style=file -i
find ./include -iname "*.h" -o -iname "*.cc" -o -iname "*.cuh" -o -iname "*.cu" | xargs clang-format-10 -style=file -i
find ./test -iname "*.h" -o -iname "*.cc" -o -iname "*.cuh" -o -iname "*.cu" | xargs clang-format-10 -style=file -i
black ./