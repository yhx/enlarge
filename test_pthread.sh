#!/bin/bash

../build.sh release double

./bin/iaf_forward_pthread 2400 10 50

../script/data_compare.py -1 /archive/share/linhui_enlarge/bsim/build/rate_cpu.IAF.log -2 /archive/share/linhui2/pattern_network/tmp/spike_count.log
