#!/bin/bash

# ./clean.sh

# ./build.sh clean

./build.sh release double

./build/bin/pattern_circle_iaf_mpi 50000

mpirun -n 16 --hostfile ./openmpi.config -mca btl_tcp_if_include eno1 ./spack_run.sh ./build/bin/pattern_circle_iaf_mpi_run

./script/column_merge.py -s rate_mpi.IAF.log

./script/line_sum.py -f rate_mpi_merge.IAF.log

