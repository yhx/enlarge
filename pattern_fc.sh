#!/bin/bash

# ../clean.sh

# ../build.sh clean

# ../build.sh release double

# 单机最大大小定为：1000神经元每层，共6个population
clear

# neuron_num=1000
# pop_num=6
run_time=10.0

# # strong scale
# for node_num in 1 2 3 4 5 6 7 8
# do
#     dir_name="strong_fc_${node_num}"
#     mkdir $dir_name
#     cd $dir_name

#     echo "START: " $node_num
#     echo "PARAMETER: " $neuron_num $(($node_num * 2))

#     # rm -rf ./pattern_circle_iaf_mpi_*
#     # ../../build/bin/pattern_fc_iaf_mpi $neuron_num $pop_num $(($node_num * 2)) > ./tmp.log 
#     mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $neuron_num $pop_num $run_time  

#     echo "FINISH: " $node_num "!"
#     cd ..
# done

# weak scale
# neuron_nums=(1000 1195 1267 1290 1290 1279 1261 1240)
neuron_nums=(1809 2335 2631 2812 2927 3000 3043 3067)
for node_num in 1 2 3 4 5 6 7 8
# for node_num in 8
do
    dir_name="weak_fc_${node_num}"
    mkdir $dir_name
    cd $dir_name

    echo "START: " $node_num "!"
    echo "Current neuron num: " ${neuron_nums[ $(($node_num - 1)) ]} "!"
    echo "Current population num: " $(($node_num + 10)) "!"
    
    cur_neuron_num=${neuron_nums[ $(($node_num - 1)) ]}
    ../../../build/bin/pattern_fc_iaf_mpi $cur_neuron_num $(($node_num + 10)) $(($node_num * 2)) > ./tmp.log 
    
    # with multi-level
    mpirun -n $node_num --hostfile ../../../openmpi1.config -mca btl_tcp_if_include eno1 ../../../spack_run.sh ../../../build/bin/pattern_fc_iaf_mpi_run $cur_neuron_num $(($node_num + 10)) $run_time

    # without multi-level
    # mpirun -n $(($node_num * 2)) --hostfile ../../../openmpi.config -mca btl_tcp_if_include eno1 ../../../spack_run.sh ../../../build/bin/pattern_fc_iaf_mpi_run $cur_neuron_num $(($node_num + 5)) $run_time
    
    echo "FINISH: " $node_num "!"
    cd ..
done