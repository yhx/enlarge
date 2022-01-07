#!/bin/bash

# ../clean.sh

# ./build.sh clean

# ./build.sh release double

rm -rf ./pattern_circle_iaf_mpi_*

# 单机模型
# for neuron_num in 15000 # 20000 30000 # 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000
# do
#     echo $neuron_num
#     ./build/bin/pattern_circle_iaf_mpi $neuron_num 2

#     # mpirun -n 8 --hostfile ./openmpi1.config -mca btl_tcp_if_include eno1 ./spack_run.sh ./build/bin/pattern_circle_iaf_mpi_run $neuron_num
#     mpirun -n 1 ./build/bin/pattern_circle_iaf_mpi_run $neuron_num

#     ./script/column_merge.py -s rate_gpu_mpi.IAF.log

#     ./script/line_sum.py -f rate_gpu_mpi_merge.IAF.log
# done

neuron_num=15000
run_time=10.0
file_name="../res.log"
echo '' > file_name

# strong scale
for node_num in 1 2 3 4 5 6 7 8
do
    dir_name="strong$node_num"
    mkdir $dir_name
    cd $dir_name

    echo "START: " $node_num
    echo "PARAMETER: " $neuron_num $(($node_num * 2))

    # rm -rf ./pattern_circle_iaf_mpi_*
    # ../../build/bin/pattern_circle_iaf_mpi $neuron_num $(($node_num * 2)) > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $neuron_num $run_time #>> $file_name

    echo "FINISH: " $node_num "!"
    cd ..
done

# weak scale
neuron_nums=(15000 21213 25981 30000 33541 36743 39687 42427)
for node_num in 2 3 4 5 6 7 8
do
    dir_name="weak$node_num$neuron_nums"
    mkdir $dir_name
    cd $dir_name

    cur_neuron_num=${neuron_nums[ $(($node_num - 1)) ]}
    echo $node_num
    echo "current neuron_num: " $cur_neuron_num
    # ../../build/bin/pattern_circle_iaf_mpi $cur_neuron_num  $(($node_num * 2)) > ./tmp.log 
    mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $cur_neuron_num $run_time #>> $file_name

    # MultiNode version
    # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $cur_neuron_num $run_time #>> $file_name

    cd ..
done
