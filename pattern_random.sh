#!/bin/bash

# ../clean.sh

# ../build.sh clean

# ../build.sh debug double

# 单机最大大小定为：1000神经元每层，共6个population
clear

neuron_num=630
run_time=1.0
node_num=8
pop_num=13

dir_name="random_${node_num}"
mkdir $dir_name
cd $dir_name

echo "START: " $node_num "!"
echo "Current neuron num: " $neuron_num "!"
echo "Current population num: " 13 "!"

../../build/bin/pattern_random_iaf_mpi $neuron_num $pop_num $(($node_num * 2)) # > ./tmp.log 
mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_random_iaf_mpi_run $neuron_num $pop_num $run_time

# MultiNodeVersion
# mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $cur_neuron_num $(($node_num + 5))

echo "FINISH: " $node_num "!"
cd ..