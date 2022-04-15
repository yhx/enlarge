#!/bin/bash

../clean.sh

../build.sh clean

../build.sh prof double

# weak scale
neuron_nums=(15000 21213 25981 30000 33541 36743 39687 42427)  # 在这个数组中写每个节点数对应的神经元数量
for node_num in 1 2 3 4 5 6 7 8
do
    dir_name="weak2_$node_num$neuron_nums"
    mkdir $dir_name
    cd $dir_name

    cur_neuron_num=${neuron_nums[ $(($node_num - 1)) ]}
    echo $node_num
    echo "current neuron_num: " $cur_neuron_num
    # ../../build/bin/pattern_circle_iaf_mpi $cur_neuron_num  $(($node_num * 2)) > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $cur_neuron_num #>> $file_name

    cd ..
done
