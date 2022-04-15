#!/bin/bash

# 现版本为神经元数量*0.2，突触数量*117

# python ./script/gain_multi_area_network.py 2a763b6a5266469124ad24b76561379d 3d9d793dc5c7ff63572664081bf8895b 16

./build.sh release double

./build/bin/construct_network 

mpirun -n 16 --hostfile ./openmpi.config -mca btl_tcp_if_include eno1 ./spack_run.sh ./build/bin/run_network 

./script/column_merge.py -s rate_gpu_mpi.IAF.log

./script/line_sum.py -f rate_gpu_mpi_merge.IAF.log