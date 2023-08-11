#!/bin/bash
# t: memory_type (0: None, 1: Continuous, 2: Chunked, 3 Distributed)
# l: memory_location (0: None, 1: Device, 2: Host)
# e: embedding_table_size (byte)
# g: gather_size (byte)
# d: embedding_dim
# c: loop_count

set -e 
#set -x

embedding_table_size=$((200*1024*1024*1024))
gatehr_size=$((4*1024*1024*1024))
loop_count=$((200))
test_type=gather
CUDA_VISIBLE_DEVICES=1 ./gbench/GATHER_SCATTER_BENCH -t 1 -l 1 -e 1024000 -g 1024 -d 32 -c 10 -f gather


for ((i=32;i<=1024;i=i*2))
do 
    ./gbench/GATHER_SCATTER_BENCH -t 2 -l 1 -e ${embedding_table_size} -g ${gatehr_size} -d ${i} -c ${loop_count} -f ${test_type}
done

for ((i=1024;i<=1024;i=i*2))
do 
    ./gbench/GATHER_SCATTER_BENCH -t 1 -l 1 -e ${embedding_table_size} -g ${gatehr_size} -d ${i} -c ${loop_count} -f ${test_type}
done

for ((i=1024;i<=1024;i=i*2))
do 
    ./gbench/GATHER_SCATTER_BENCH -t 1 -l 2 -e ${embedding_table_size} -g ${gatehr_size} -d ${i} -c ${loop_count} -f ${test_type}
done
