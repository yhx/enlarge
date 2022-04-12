#!/bin/bash

# ./build.sh release double
clear

for i in 1 2 3 4 5 6 7 8 9 10
do
    echo 'CURRENT ROUND: ' $i ' START' 
    forward_file="../run_all/forward_${i}.log"
    circle_file="../run_all/circle_${i}.log"
    fc_file="../run_all/fc_${i}.log"
    multi_delay_file="../run_all/multi_delay_${i}.log"
    multi_area_file="./run_all/multi_area_${i}.log"
    multi_area_file2="./run_all/multi_area2_${i}.log"

    cd ./scale_forward
    ../pattern_forward.sh >& $forward_file
    cd ..

    cd ./scale_circle
    ../pattern_circle.sh >& $circle_file
    cd ..

    cd ./scale_fc
    ../pattern_fc.sh >& $fc_file
    cd ..

    ./run_multi_area_model_multi_level.sh >& $multi_area_file

    ./run_mutli_area_model.sh >& $multi_area_file2

    echo 'CURRENT ROUND: ' $i ' END'
done 