#!/bin/bash

if [[ $# -ne 3 ]]; then
	echo "Usuage: to_metis.sh inputfile nparts grp_size"
	exit -1
fi

name=${1%.*}

echo $name

echo "/archive/share/metis/to_metis_grp $1 $name.grp $3"
/archive/share/metis/to_metis_grp $1 $name.grp $3
echo "/archive/share/metis/to_metis_s2 $name.grp $name.grp2"
/archive/share/metis/to_metis_s2 $name.grp $name.grp2
echo "/archive/share/metis/to_metis_s3 $name.grp2 $name.grp3 $2"
/archive/share/metis/to_metis_s3 $name.grp2 $name.grp3 $2
echo "/archive/share/metis/to_metis_grp4 $1 $name.grp3 $name.grp.metis $3"
/archive/share/metis/to_metis_grp4 $1 $name.grp3 $name.grp.metis $3
