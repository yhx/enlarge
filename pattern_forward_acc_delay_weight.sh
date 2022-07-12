#!/bin/bash


# neuron_num=1000
# depth=3500
# run_time=10.0
# node_num=8

neuron_num=1000
depth=3000
run_time=10.0
node_num=8
delay_num=10

# dir_name="forward_acc_test_3500_delay5"
dir_name="forward_acc_test_3500_delay_weight"
mkdir $dir_name
cd $dir_name
rm -rf ./*.log
rm -rf ./sum.res

# ../../build/bin/pattern_forward_iaf_mpi_delay_weight $depth $neuron_num $(($node_num * 2)) $delay_num #> ./tmp.log 

# Multi-node
rm -rf ./*.log
rm -rf ./sum.res
mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $depth $neuron_num $run_time 0
../../script/column_merge.py -s rate_gpu_mpi.IAF.log
../../script/line_sum.py -f rate_gpu_mpi_merge.IAF.log
cat ./sum.res

# Multi-level
mpirun -n $node_num --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $depth $neuron_num $run_time 1
# single GPU
# ../build/bin/pattern_forward_iaf_gpu $depth $neuron_num $run_time
../../script/column_merge.py -s rate_gpu_mpi.IAF.log
../../script/line_sum.py -f rate_gpu_mpi_merge.IAF.log
cat ./sum.res


