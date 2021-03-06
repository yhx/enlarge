#!/bin/bash
SCRIPT_PATH=`dirname "$0"`

MODE=$(echo $1 | tr [A-Z] [a-z]) 

PREC=$(echo $2 | tr [A-Z] [a-z])

THREAD_NUM=$3 

C_MODE="Release"
USE_DOUBLE="OFF"
USE_LOG="OFF"
VERBOSE=0

TOTAL_THREAD_NUM=`getconf _NPROCESSORS_ONLN`
if [ x"$THREAD_NUM" = x ]; then
	((THREAD_NUM=TOTAL_THREAD_NUM/2))
fi

if [ "$MODE" = "debug" ]; then
	C_MODE="Debug"
	VERBOSE=1
	USE_PROF="ON"
	USE_LOG="ON"
elif [ "$MODE" = "log" ]; then
	C_MODE="Release"
	VERBOSE=1
	USE_PROF="ON"
	USE_LOG="ON"
elif [ "$MODE" = "test" ]; then
	C_MODE="Debug"
	USE_LOG="ON"
	USE_PROF="ON"
	USE_DOUBLE="ON"
	VERBOSE=1
	# THREAD_NUM=1
elif [ "$MODE" = "prof" ]; then
	C_MODE="Release"
	USE_LOG="OFF"
	VERBOSE=0
	USE_PROF="ON"
	USE_LOG="OFF"
	# THREAD_NUM=1
else
	C_MODE="Release"
	USE_LOG="OFF"
	USE_PROF="OFF"
	VERBOSE=0
fi


if [ "$PREC" = "double" ]; then
	USE_DOUBLE="ON"
fi

if [ ! -d $SCRIPT_PATH/build ]; then
	mkdir $SCRIPT_PATH/build
fi


set -x
if [ "$MODE" = "clean" ]; then
	cd $SCRIPT_PATH/build && make clean-all
elif [ "$MODE" = "test" ]; then
	cd $SCRIPT_PATH/build && cmake -DCMAKE_BUILD_TYPE="Debug" -DUSE_DOUBLE=$USE_DOUBLE -DUSE_LOG=$USE_LOG -DUSE_PROF=$USE_PROF -lpthread .. 2> >(tee error.err) && make -j$THREAD_NUM VERBOSE=$VERBOSE 2> >(tee -a error.err) && make test 
else
	cd $SCRIPT_PATH/build && cmake -DCMAKE_BUILD_TYPE=$C_MODE -DUSE_DOUBLE=$USE_DOUBLE -DUSE_LOG=$USE_LOG -DUSE_PROF=$USE_PROF -lpthread .. 2> >(tee error.err) && make -j$THREAD_NUM VERBOSE=$VERBOSE 2> >(tee -a error.err)
fi
