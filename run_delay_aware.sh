#!/bin/bash

# ./build.sh release double
clear

for i in 1 2 3 4 5 6 7 8 9 10
do
    echo 'CURRENT ROUND: ' $i ' START' 
    multi_delay_file="../run_all/multi_delay_${i}.log"

    cd ./test_multi_delay
    ../pattern_multi_delay.sh >& $multi_delay_file
    cd ..

    echo 'CURRENT ROUND: ' $i ' END'
done 