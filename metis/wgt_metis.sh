#!/bin/bash

if [[ $# -ne 3 ]]; then
	echo "Usuage: to_metis.sh inputfile nparts weightfile"
	exit -1
fi

name=${1%.*}

echo $name

echo "/archive/share/metis/to_metis_s1 $1 $name.s1"
/archive/share/metis/to_metis_s1 $1 $name.s1
echo "/archive/share/metis/to_metis_s2 $name.s1 $name.s2"
/archive/share/metis/to_metis_s2 $name.s1 $name.s2
echo "/archive/share/metis/to_metis_s3 $name.s2 $name.weight.metis $2 $name.weight"
/archive/share/metis/to_metis_s3 $name.s2 $name.weight.metis $2 $name.weight
