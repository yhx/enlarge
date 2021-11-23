#!/bin/bash

./clean.sh

./build.sh release double

./build/bin/construct_network

mpirun -n 16 --hostfile ./openmpi.config -mca btl_tcp_if_include eno1 ./spack_run.sh ./build/bin/run_network
 