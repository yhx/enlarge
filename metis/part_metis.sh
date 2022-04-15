#!/bin/bash

if [[ $# -ne 3 ]]; then
	echo "Usuage: to_metis.sh inputfile nparts1 nparts2"
	exit -1
fi

name=${1%.*}

inputfile=$1
nparts1=$2
nparts2=$3

echo $name

echo "/archive/share/metis/to_metis_s1 $inputfile $name.s1"
/archive/share/metis/to_metis_s1 $inputfile $name.s1
echo "/archive/share/metis/to_metis_s2 $name.s1 $name.s2"
/archive/share/metis/to_metis_s2 $name.s1 $name.s2
echo "/archive/share/metis/to_metis_s3 $name.s2 $name.metis.org $nparts1"
/archive/share/metis/to_metis_s3 $name.s2 $name.metis.org $nparts1
echo "/archive/share/metis/part $name.s1 $name.metis.org $nparts1"
/archive/share/metis/part $name.s1 $name.metis.org $nparts1 

for ((i=0; i<=$nparts1-1; i++))
do
	echo "/archive/share/metis/to_metis_s2 $name.s1_sub$i $name.s2_sub$i"
	/archive/share/metis/to_metis_s2 $name.s1_sub$i $name.s2_sub$i
	echo "/archive/share/metis/to_metis_s3 $name.s2_sub$i $name.metis.sub_$i $nparts2"
	/archive/share/metis/to_metis_s3 $name.s2_sub$i $name.metis.sub_$i $nparts2
done

echo "/archive/share/metis/repart $name.s1 $name.metis.org $name.metis.sub $name.part.metis $nparts1 $nparts2"
/archive/share/metis/repart $name.s1 $name.metis.org $name.metis.sub $name.part.metis $nparts1 $nparts2

