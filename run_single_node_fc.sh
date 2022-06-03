#!/bin/bash

run_time=10.0
neuron_nums=(1809 2335 2631 2812 2927 3000 3043 3067)

node_num=1

dir_name="single_fc"
mkdir $dir_name
cd $dir_name

echo "START: " $node_num "!"
echo "Current neuron num: " ${neuron_nums[ $(($node_num - 1)) ]} "!"
echo "Current population num: " $(($node_num + 10)) "!"

cur_neuron_num=${neuron_nums[ $(($node_num - 1)) ]}
../../build/bin/pattern_fc_iaf_mpi $cur_neuron_num $(($node_num + 10)) $(($node_num * 2)) > ./tmp.log 
mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $cur_neuron_num $(($node_num + 10)) $run_time

# ../../build/bin/pattern_fc_gpu $cur_neuron_num $(($node_num + 10))

echo "FINISH: " $node_num "!"
cd ..
