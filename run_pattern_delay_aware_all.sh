#!/bin/bash

echo "pattern_forward_delay_aware"
../pattern_forward_delay_aware.sh #> pattern_forward_delay.txt

echo "pattern_circle_delay_aware"
../pattern_circle_delay_aware.sh #> pattern_circle_delay.txt

echo "pattern_fc_delay_aware"
../pattern_fc_delay_aware.sh #> pattern_fc_delay.txt
