#!/bin/bash


# neuron_num=1240
# pop_num=13
neuron_num=3
pop_num=5000
run_time=10.0
node_num=8

dir_name="fc_acc_test_large"
mkdir $dir_name
cd $dir_name
# rm -rf ./*.log
# rm -rf ./sum.res


# ../build/bin/pattern_fc_iaf_mpi $neuron_num $pop_num $(($node_num * 2)) > ./tmp.log 
echo "finish building network!"


# multi-node
mpirun -n $(($node_num * 2)) --hostfile ../openmpi.config -mca btl_tcp_if_include eno1 ../spack_run.sh ../build/bin/pattern_fc_iaf_mpi_run $neuron_num $pop_num $run_time 0
../script/column_merge.py -s rate_gpu_mpi.IAF.log
../script/line_sum.py -f rate_gpu_mpi_merge.IAF.log
cat ./sum.res


# multi-level
mpirun -n $node_num --hostfile ../openmpi1.config -mca btl_tcp_if_include eno1 ../spack_run.sh ../build/bin/pattern_fc_iaf_mpi_run $neuron_num $pop_num $run_time 1
# single GPU
# ../build/bin/pattern_forward_iaf_gpu $depth $neuron_num $run_time
../script/column_merge.py -s rate_gpu_mpi.IAF.log
../script/line_sum.py -f rate_gpu_mpi_merge.IAF.log
cat ./sum.res
