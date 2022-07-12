#!/bin/bash


neuron_num=1000
depth=2400
run_time=10.0
node_num=8

dir_name="forward_acc_test_delay_all"
mkdir $dir_name
cd $dir_name
# rm -rf ./*.log
# rm -rf ./sum.res

delay_num=500

mkdir delay_$delay_num
cd delay_$delay_num

# ../../../build/bin/pattern_forward_iaf_mpi_delay $depth $neuron_num $(($node_num * 2)) $delay_num > ./tmp.log 
echo "finish building network!"

mpirun -n $node_num --hostfile ../../../openmpi1.config -mca btl_tcp_if_include eno1 ../../../spack_run.sh ../../../build/bin/pattern_forward_iaf_mpi_run $depth $neuron_num $run_time 1

cd ..

cd ..