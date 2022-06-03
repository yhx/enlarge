#!/bin/bash

# ../clean.sh

# ../build.sh clean

# ../build.sh release double

# 单机最大大小定为：1500神经元每层，共300层

# 单机模型
# depth=300
# rm -rf ./pattern_forward_iaf_mpi_*
# for neuron_num in 1000 1500 2000 #2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000 15000 20000 30000
# do
#     echo $neuron_num
#     ../build/bin/pattern_forward_iaf_mpi $depth $neuron_num 2

#     # mpirun -n 8 --hostfile ./openmpi1.config -mca btl_tcp_if_include eno1 ./spack_run.sh ./build/bin/pattern_forward_iaf_mpi_run $neuron_num
#     mpirun -n 1 ../build/bin/pattern_forward_iaf_mpi_run $depth $neuron_num
# done

neuron_num=1000
depth=300
run_time=10.0

# strong scale
# for node_num in 1 2 3 4 5 6 7 8
# do
#     dir_name="strong_forward_$node_num"
#     mkdir $dir_name
#     cd $dir_name
#     echo "runtime: " $run_time
#     echo "START: " $node_num
#     echo "PARAMETER: " $neuron_num $(($node_num * 2))

#     # rm -rf ./pattern_circle_iaf_mpi_*
#     # ../../build/bin/pattern_forward_iaf_mpi $depth $neuron_num $(($node_num * 2)) > ./tmp.log 
#     mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $depth $neuron_num $run_time

#     echo "FINISH: " $node_num "!"
#     cd ..
# done

# weak scale
for node_num in 2 3 4 5 6 7 8
do
    dir_name="weak_forward_$node_num"
    mkdir $dir_name
    cd $dir_name

    echo "START: " $node_num "!"
    
    ../../../build/bin/pattern_forward_iaf_mpi $(($depth * $node_num)) $neuron_num  $(($node_num * 2)) > ./tmp.log 
    
    # with multi-level
    mpirun -n $node_num --hostfile ../../../openmpi1.config -mca btl_tcp_if_include eno1 ../../../spack_run.sh ../../../build/bin/pattern_forward_iaf_mpi_run $(($depth * $node_num)) $neuron_num $run_time
    
    # without multi-level
    # mpirun -n $(($node_num * 2)) --hostfile ../../../openmpi.config -mca btl_tcp_if_include eno1 ../../../spack_run.sh ../../../build/bin/pattern_forward_iaf_mpi_run $(($depth * $node_num)) $neuron_num $run_time
    
    echo "FINISH: " $node_num "!"
    cd ..
done

