#!/bin/bash

# ./build.sh release double

node_num=8


# ../build/bin/construct_network $(($node_num * 2))

mpirun -n $node_num --hostfile ./openmpi1.config -mca btl_tcp_if_include eno1 ./spack_run.sh ./build/bin/run_network_multi_level

# calculate rate and fire count
# ./script/column_merge.py -s rate_gpu_mpi.IAF.log

# ./script/line_sum.py -f rate_gpu_mpi_merge.IAF.log
