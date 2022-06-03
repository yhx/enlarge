#!/bin/bash

neuron_num=15000
run_time=10.0

neuron_nums=(15000 21213 25981 30000 33541 36743 39687 42427)
node_num=1

dir_name="single_circle"
mkdir $dir_name
cd $dir_name

cur_neuron_num=${neuron_nums[ $(($node_num - 1)) ]}
echo $node_num
echo "current neuron_num: " $cur_neuron_num
../../build/bin/pattern_circle_iaf_mpi $cur_neuron_num  $(($node_num * 2)) > ./tmp.log 
mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $cur_neuron_num $run_time #>> $file_name

# ../../build/bin/pattern_circle_gpu $cur_neuron_num 

cd ..