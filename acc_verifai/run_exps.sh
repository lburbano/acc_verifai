#!/bin/bash

echo "Run all experiments" 
for j in {11..15}
do
    python falsifier.py --model carla --path "scenarios/platv1a-dist$j.scenic" --e "setpoint-$j" --route "res-$j" 
    echo "Finished distance $j" 
done
echo "Remember to change sampler for the next one!"
