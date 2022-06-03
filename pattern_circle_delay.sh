#!/bin/bash


neuron_num=42427
run_time=10.0
node_num=8
delay_num=500

dir_name="circle_acc_test_delay_$delay_num"
mkdir $dir_name
cd $dir_name
# rm -rf ./*.log
# rm -rf ./sum.res


../../build/bin/pattern_circle_iaf_mpi_delay $neuron_num $(($node_num * 2)) $delay_num > ./tmp.log 
mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $neuron_num $run_time 
