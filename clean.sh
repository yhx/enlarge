#!/bin/sh

EXT="bak data head log csv count info res send recv cs cm map"
DIR="build build/bin build/bin/backup backup ."
# SPT_DIR=$(dirname $(readlink -f "$0"))
# echo $SPT_DIR

for dir in $DIR
do
	for ext in $EXT
	do
		if [ -d "$PWD/$dir" ]; then
			# echo "remove file extent $ext in $PWD/$dir"
			rm -fv $PWD/$dir/*.$ext
		fi
	done
done

# rm ./*.bak
# rm ./*.data
# rm ./*.head
# rm ./*.log
# rm ./*.csv
# rm ./*.count
# rm ./*.info
# rm ./*.res
# rm ./*.send
# rm ./*.recv
# rm ./*.cs
# rm ./*.cm
# rm ./*.map
# rm ./build/*/*.bak
# rm ./build/*/*.data
# rm ./build/*/*.head
# rm ./build/*/*.log
# rm ./build/*/*.csv
# rm ./build/*/*.count
# rm ./build/*/*.info
# rm ./build/*/*.res
# rm ./build/*/*.send
# rm ./build/*/*.recv
# rm ./build/*/*.cs
# rm ./build/*/*.cm
# rm ./build/*/*.map
# rm ./build/*/*/*.bak
# rm ./build/*/*/*.data
# rm ./build/*/*/*.head
# rm ./build/*/*/*.log
# rm ./build/*/*/*.csv
# rm ./build/*/*/*.count
# rm ./build/*/*/*.info
# rm ./build/*/*/*.res
# rm ./build/*/*/*.send
# rm ./build/*/*/*.recv
# rm ./build/*/*/*.cs
# rm ./build/*/*/*.cm
# rm ./build/*/*/*.map
