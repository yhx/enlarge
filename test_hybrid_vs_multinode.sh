#!/bin/bash

if [ ! -d "./hybrid" ];then
    mkdir ./hybrid
fi

if [ ! -d "./multinode" ];then
    mkdir ./multinode
fi

depth=1000
neuron_num=100
part=16
delay=100

thread_num=3
gpu_num=2

cd multinode

mpirun -n 16 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../bin/pattern_forward_multinode_simulator $depth $neuron_num $part $delay  

../../script/column_merge.py -s rate_gpu_mpi.IAF.log

../../script/line_sum.py -f rate_gpu_mpi_merge.IAF.log

cd ..


cd hybrid

mpirun -n 16 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../bin/pattern_forward_hybrid_simulator $depth $neuron_num $part $delay $thread_num $gpu_num  

../../script/column_merge.py -s rate_gpu_mpi.IAF.log

../../script/line_sum.py -f rate_gpu_mpi_merge.IAF.log

../../script/column_merge.py -s rate_cpu_mpi.IAF.log -i rate_gpu_mpi_merge.IAF.log

../../script/line_sum.py -f rate_cpu_mpi_merge.IAF.log

cd ..
