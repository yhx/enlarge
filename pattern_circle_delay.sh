#!/bin/bash


neuron_num=42427
run_time=10.0
node_num=8

dir_name="circle_acc_test_delay"
mkdir $dir_name
cd $dir_name
# rm -rf ./*.log
# rm -rf ./sum.res

for delay_num in 1 2 4 8 
do
mkdir delay_$delay_num
cd delay_$delay_num

# ../../../build/bin/pattern_circle_iaf_mpi_delay $neuron_num $(($node_num * 2)) $delay_num > ./tmp.log 
echo "finish building network!"

mpirun -n $node_num --hostfile ../../../openmpi1.config -mca btl_tcp_if_include eno1 ../../../spack_run.sh ../../../build/bin/pattern_circle_iaf_mpi_run $neuron_num $run_time 1

../../../script/column_merge.py -s rate_gpu_mpi.IAF.log
../../../script/line_sum.py -f rate_gpu_mpi_merge.IAF.log
cat ./sum.res

cd ..
done

cd ..
