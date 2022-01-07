clear

# to run:
# 1. create a directory like "mkdir test_pangmeng"
# 2. then "cd test_pangmeng" (program should run in a directory)
# 3. run "../pattern_pangmeng.sh"

# modify: 
# 1. just replace "50 100" to neuron numbers list needed to test

# all programs run with node number equals to 8 with MultiLevelSimulator

# ../build.sh prof double

# circle
file_name="../log"
node_num=1
run_time=10.0
echo "" > "./log"
# for neuron_num in 15000 #1875 3750 5625 7500 9375 11250 13125 15000
# do
#     dir_name="circle_${node_num}_${neuron_num}"
#     mkdir $dir_name
#     cd $dir_name

#     # echo $node_num
#     echo "======================================neuron_num: " $neuron_num
#     echo "======================================neuron_num: " $neuron_num >> $file_name
#     # ../../build/bin/pattern_circle_iaf_mpi $neuron_num  $(($node_num * 2)) > ./tmp.log 
#     ../../build/bin/pattern_circle_iaf_gpu $neuron_num $run_time >> $file_name

#     # MultiNode version
#     # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $cur_neuron_num #>> $file_name

#     cd ..
# done

# forward
# node_num=1
# depth=300
# for neuron_num in 125 250 375 500 625 750 875 1000
# do
#     dir_name="forward_${neuron_num}"
#     mkdir $dir_name
#     cd $dir_name

#     echo "======================================neuron_num: " $neuron_num
#     echo "======================================neuron_num: " $neuron_num >> $file_name
    
#     # ../../build/bin/pattern_forward_iaf_mpi $(($depth * $node_num)) $neuron_num  $(($node_num * 2)) > ./tmp.log 
#     ../../build/bin/pattern_forward_iaf_gpu $(($depth * $node_num)) $neuron_num $run_time >> $file_name
    
#     # multi node version
#     # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $(($depth * $node_num)) $neuron_num
#     cd ..
# done

# fc
node_num=1
for neuron_num in 125 250 375 500 625 750 875 1000
do
    dir_name="fc_${neuron_num}_${node_num}"
    # mkdir $dir_name
    cd $dir_name

    echo "======================================neuron_num: " $neuron_num
    echo "======================================neuron_num: " $neuron_num >> $file_name
    # ../../build/bin/pattern_fc_iaf_mpi $neuron_num $(($node_num + 5)) $(($node_num * 2)) > ./tmp.log 
    ../../build/bin/pattern_fc_iaf_gpu $neuron_num $(($node_num + 5)) $run_time # >> $file_name

    # MultiNodeVersion
    # mpirun -n $(($node_num * 2)) --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $cur_neuron_num $(($node_num + 5))
    
    cd ..
done
