#!/bin/bash


# neuron_num=1240
# pop_num=13
neuron_num=3067
pop_num=18
run_time=10.0
node_num=8
delay_num=500

dir_name="fc_acc_test_delay_$delay_num"
mkdir $dir_name
cd $dir_name
# rm -rf ./*.log
# rm -rf ./sum.res


# ../../build/bin/pattern_fc_iaf_mpi_delay $neuron_num $pop_num $(($node_num * 2)) $delay_num > ./tmp.log 
echo "finish building network!"
mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $neuron_num $pop_num $run_time  

