#!/bin/bash

if [[ $# -ne 2 ]]; then
	echo "Usuage: to_metis.sh inputfile nparts"
	exit -1
fi

name=${1%.*}
echo $name

echo "/archive/share/pm20/bsim_test/metis/to_metis_s1 $1 $name.s1"
/archive/share/pm20/bsim_test/metis/to_metis_s1 $1 $name.s1
echo "/archive/share/pm20/bsim_test/metis/to_metis_s2 $name.s1 $name.s2"
/archive/share/pm20/bsim_test/metis/to_metis_s2 $name.s1 $name.s2
echo "/archive/share/pm20/bsim_test/metis/to_metis_s3 $name.s2 $name.metis $2"
/archive/share/pm20/bsim_test/metis/to_metis_s3 $name.s2 $name.metis $2
