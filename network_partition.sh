clear


##pattern cycle
file_name="../result.log"
rm metis_opt.log



neuron_num=42427
# for split in 1 2 3 8 
for split in 1 2 3 7
do
    dir_name="gs_cicle_"
    mkdir $dir_name$split
    cd $dir_name$split
    
    # split=7
    # if [ $split == 8 ]; then
    #     split=7
    # fi
    split=7
    echo "pattern cycle=================================" $split
    echo "pattern cycle=================================" $split >> $file_name
    # rm -rf ./pattern_circle_iaf_mpi_*
    if [ ! -d pattern_circle_iaf_mpi_42427 ]; then
        echo "begin build..."
        ../../build/bin/pattern_circle_iaf_mpi $neuron_num 16 $split > ./tmp.log 
    fi
    mpirun -n 8 --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $neuron_num >> $file_name

    echo "FINISH: " $split "!"
    cd ..     
done

#fc
neuron_num=1240
pop_num=13
# for split in 1 2 3 8
for split in 1 2 3 7
do
    dir_name="gs_fc_"
    mkdir $dir_name$split
    cd $dir_name$split

    # if [ $split == 8 ]; then
    #     split=7
    # fi
    split=7
    echo "pattern fc======================================" $split
    echo "pattern fc======================================" $split >> $file_name

    #  if [ $split == "5" ]; then
    #     split=0
    # fi
    # split=7
    # rm -rf ./pattern_fc_iaf_mpi_*
    if [ ! -d pattern_fc_iaf_mpi_1240_13 ]; then
        echo "begin build..."
        ../../build/bin/pattern_fc_iaf_mpi $neuron_num $pop_num 16 $split > ./tmp.log 
    fi
    mpirun -n 8 --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $neuron_num $pop_num >> $file_name

    echo "FINISH: " $split "!"
    cd ..
done

## forwd
neuron_num=1000
depth=2400
# for split in 1 2 3 8 
for split in 1 2 3 7
do
    dir_name="gs_forwd_"
    mkdir $dir_name$split
    cd $dir_name$split
    # if [ $split == 8 ]; then
    #     split=7
    # fi

    split=7

    echo "pattern forwd======================================" $split
    echo "pattern forwd======================================" $split >> $file_name

    # rm -rf ./pattern_forward_iaf_mpi_*
    # split=7
    if [ ! -d pattern_forward_iaf_mpi_2400_1000 ]; then
        echo "begin build..."
        ../../build/bin/pattern_forward_iaf_mpi $depth $neuron_num 16 $split > ./tmp.log 
    fi
    mpirun -n 8 --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $depth $neuron_num >> $file_name

    echo "FINISH: " $split "!"
    cd ..
done



#  ### random
neuron_num=50
run_time=10.0
node_num=8
pop_num=500

for split in 1 2 3 7
do 
    dir_name="gs_Random_"
    mkdir $dir_name$split
    cd $dir_name$split

    echo "pattern random======================================" $split
    echo "pattern random======================================" $split >> $file_name

    # if [ $split == "5" ]; then
    #     split=0
    # fi

    if [ ! -d pattern_random_iaf_mpi_"$neuron_num"_"$pop_num" ]; then
        echo "begin build...  pattern_random_iaf_mpi_"$neuron_num"_"$pop_num" "
        ../../build/bin/pattern_random_iaf_mpi2 $neuron_num $pop_num 16 $split > ./tmp.log 
    fi
    mpirun -n 8 --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_random_iaf_mpi_run $neuron_num $pop_num $run_time >> $file_name
    # mpirun -n 16 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_random_iaf_mpi_run $neuron_num $pop_num $run_time >> $file_name

    echo "FINISH: " $split "!"
    cd ..
done 


#for split in 1 2 3 7
for split in 1 2 3 7
do

    echo "mvc...."
    echo $split
    dir_name="gs_mvc_"
    mkdir $dir_name$split
    cd $dir_name$split

    if [ ! -d multi_area_model_20_117 ]; then
        echo "begin build..."
        ../../build/bin/construct_network $split > tmp.log
    fi 

    mpirun -n 8 --hostfile ../../openmpi1.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/run_network_multi_level >> $file_name
    cd ..
done 
