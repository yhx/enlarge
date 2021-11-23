#!/bin/bash

./build.sh release double

./build/bin/construct_network 

mpirun -n 16 --hostfile ./openmpi.config -mca btl_tcp_if_include eno1 ./spack_run.sh ./build/bin/run_network 

./script/column_merge.py -s rate_gpu_mpi.IAF.log

./script/line_sum.py -f rate_gpu_mpi_merge.IAF.log
