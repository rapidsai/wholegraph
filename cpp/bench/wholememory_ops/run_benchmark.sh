#!/bin/bash
# t: memory_type (0: None, 1: Continuous, 2: Chunked, 3 Distributed)
# l: memory_location (0: None, 1: Device, 2: Host)
# e: embedding_table_size (byte)
# g: gather_size (byte)
# d: embedding_dim
# c: loop_count

set -e 
set -x

embedding_table_size=$((200*1024*1024*1024))
gatehr_size=$((4*1024*1024*1024))
loop_count=$((200))
test_type=gather

for ((embedding_dim=1024;embedding_dim<=1024;embedding_dim*=2))
do 
    ./gbench/GATHER_SCATTER_BENCH -t 2 -l 1 -e ${embedding_table_size} -g ${gatehr_size} -d ${embedding_dim} -c ${loop_count} -f ${test_type}
done

for ((embedding_dim=1024;embedding_dim<=1024;embedding_dim*=2))
do 
    ./gbench/GATHER_SCATTER_BENCH -t 1 -l 1 -e ${embedding_table_size} -g ${gatehr_size} -d ${embedding_dim} -c ${loop_count} -f ${test_type}
done

for ((embedding_dim=1024;embedding_dim<=1024;embedding_dim*=2))
do 
    ./gbench/GATHER_SCATTER_BENCH -t 1 -l 2 -e ${embedding_table_size} -g ${gatehr_size} -d ${embedding_dim} -c ${loop_count} -f ${test_type}
done
