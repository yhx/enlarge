#!/bin/bash

./build.sh log double

./build/bin/pattern_multi_delay 2

python ./script/convert_fire_time.py
