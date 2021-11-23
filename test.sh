#!/bin/bash

# remove exist files
rm ./bak/gpu/*
rm ./bak/ml/*

depth=1000;
num_neuron=10;
fire_rate=5;
N=8;
delay=8;


echo depth num_neuron fire_rate N delay

######### build network and save
./build/bin/forward_mpi_gen ${depth} ${num_neuron} ${fire_rate} ${delay} ${N}

######### run forward_mpi_gpu
mpirun -n ${N} ./build/bin/forward_mpi_gpu2 ${depth} ${num_neuron} ${fire_rate} ${delay}

mv *.log ./bak/gpu

echo "" > ./bak/gpu/rate_gpu_mpi.log

python3 ./script/column_merge.py -o ./bak/rate_gpu_mpi_gpu.log -i ./bak/gpu/rate_gpu_mpi_0.LIF.log -i ./bak/gpu/rate_gpu_mpi_1.LIF.log -i ./bak/gpu/rate_gpu_mpi_2.LIF.log -i ./bak/gpu/rate_gpu_mpi_3.LIF.log -i ./bak/gpu/rate_gpu_mpi_4.LIF.log -i ./bak/gpu/rate_gpu_mpi_5.LIF.log -i ./bak/gpu/rate_gpu_mpi_6.LIF.log -i ./bak/gpu/rate_gpu_mpi_7.LIF.log


######### run forward_mpi_ml
mpirun -n ${N} ./build/bin/forward_mpi_ml2 ${depth} ${num_neuron} ${fire_rate} ${delay}

mv *.log ./bak/ml

echo "" > ./bak/ml/rate_gpu_mpi.log

python3 ./script/column_merge.py -o ./bak/rate_gpu_mpi_ml.log -i ./bak/ml/rate_gpu_mpi_0.LIF.log -i ./bak/ml/rate_gpu_mpi_1.LIF.log -i ./bak/ml/rate_gpu_mpi_2.LIF.log -i ./bak/ml/rate_gpu_mpi_3.LIF.log -i ./bak/ml/rate_gpu_mpi_4.LIF.log -i ./bak/ml/rate_gpu_mpi_5.LIF.log -i ./bak/ml/rate_gpu_mpi_6.LIF.log -i ./bak/ml/rate_gpu_mpi_7.LIF.log

python3 ./script/data_compare.py -1 ./bak/rate_gpu_mpi_gpu.log -2 ./bak/rate_gpu_mpi_ml.log
