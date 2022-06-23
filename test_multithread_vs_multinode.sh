#!/bin/bash

rm -rf ./pthread
rm -rf ./multinode

if [ ! -d "./pthread" ];then
    mkdir ./pthread
fi

if [ ! -d "./multinode" ];then
    mkdir ./multinode
fi

depth=1000
neuron_num=100
part=16
delay=100

thread_num=28

cd multinode

mpirun -n 2 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../bin/pattern_forward_multinode_simulator $depth $neuron_num $part $delay  

../../script/column_merge.py -s rate_gpu_mpi.IAF.log

../../script/line_sum.py -f rate_gpu_mpi_merge.IAF.log

cd ..

cd pthread

../bin/pattern_forward_pthread $depth $neuron_num $part $delay $thread_num

../../script/line_sum.py -f rate_cpu.IAF.log

cd ..

echo "multinode:"
cat ./multinode/sum.res

echo "pthread:"
cat ./pthread/sum.res
