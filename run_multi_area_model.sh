#!/bin/bash

# ./build.sh release double

# ./build/bin/construct_network 

cd multi_area_30_16_8/
# rm -rf ./*.log
# rm -rf ./sum.res
# mpirun -n 16 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/run_network 
# ../../script/column_merge.py -s rate_gpu_mpi.IAF.log
# ../../script/line_sum.py -f rate_gpu_mpi_merge.IAF.log
# cat ./sum.res

rm -rf ./*.log
rm -rf ./sum.res
mpirun -n 8 --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/run_network_multi_level
../../script/column_merge.py -s rate_gpu_mpi.IAF.log
../../script/line_sum.py -f rate_gpu_mpi_merge.IAF.log
cat ./sum.res

cd ..
