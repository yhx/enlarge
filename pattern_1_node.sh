cd ./scale_forward
neuron_num=1000
depth=300
# strong scale
node_num=1
dir_name="strong_forward_$node_num"
mkdir $dir_name
cd $dir_name

echo "START: " $node_num
echo "PARAMETER: " $neuron_num $(($node_num * 2))

# rm -rf ./pattern_circle_iaf_mpi_*
# ../../build/bin/pattern_forward_iaf_mpi $depth $neuron_num $(($node_num * 2)) > ./tmp.log 
mpirun -n 2 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_forward_iaf_mpi_run $depth $neuron_num 

echo "FINISH: " $node_num "!"
cd ../../

cd ./scale_circle
neuron_num=15000
file_name="../res.log"
echo '' > file_name
node_num=1
dir_name="strong$node_num"
mkdir $dir_name
cd $dir_name

echo "START: " $node_num
echo "PARAMETER: " $neuron_num $(($node_num * 2))

# rm -rf ./pattern_circle_iaf_mpi_*
# ../../build/bin/pattern_circle_iaf_mpi $neuron_num $(($node_num * 2)) > ./tmp.log 
mpirun -n 2 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_circle_iaf_mpi_run $neuron_num #>> $file_name

echo "FINISH: " $node_num "!"
cd ../../

cd ./scale_fc
neuron_num=1000
pop_num=6
node_num=1
dir_name="strong_fc_${node_num}"
mkdir $dir_name
cd $dir_name

echo "START: " $node_num
echo "PARAMETER: " $neuron_num $(($node_num * 2))

# rm -rf ./pattern_circle_iaf_mpi_*
# ../../build/bin/pattern_fc_iaf_mpi $neuron_num $pop_num $(($node_num * 2)) > ./tmp.log 
mpirun -n 2 --hostfile ../../openmpi.config -mca btl_tcp_if_include eno1 ../../spack_run.sh ../../build/bin/pattern_fc_iaf_mpi_run $neuron_num $pop_num  

echo "FINISH: " $node_num "!"
cd ../../

