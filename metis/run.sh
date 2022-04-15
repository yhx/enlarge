name=${1%.*}
inputfile=$1


echo $name

echo "to_metis_all $inputfile $name.metis"
/archive/share/pm20/bsim_test/metis/to_metis_all $inputfile $name.metis 8 128 2
