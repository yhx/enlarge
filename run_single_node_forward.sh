#!/bin/bash

neuron_num=1000
depth=300
run_time=10.0

node_num=1
dir_name="single_forward"
mkdir $dir_name
cd $dir_name

echo "START: single node forward!"

# ../../build/bin/pattern_forward_iaf_mpi $(($depth * $node_num)) $neuron_num  $(($node_num * 2)) > ./tmp.log 
mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $(($depth * $node_num)) $neuron_num $run_time 1

# ../../build/bin/pattern_forward_gpu $(($depth * $node_num)) $neuron_num

echo "FINISH: single node forward!"
cd ..
