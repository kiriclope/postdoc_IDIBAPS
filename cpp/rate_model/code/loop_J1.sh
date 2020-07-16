#!/usr/bin/env bash
LANG=en_US
export LANG

read dir n_pop N K <<<  "$1 $2 $3 $4" 
read pMin dp pMax <<< "$5 ${6} ${7}"
	
for i in $(seq 0.0 0.1 1.0); do 
    
    sed -ie "s/MEAN_XI .*/MEAN_XI $i/" globals.h 
    echo "MEAN_XI_$i"
    
    sleep 1 
    echo "Compiling ..."
    g++ main.cpp -std=c++11 -Ofast -s -o loop_sigma.out
    sleep 2
    
    screen -dmS  N${N}K${K}_MEAN_XI_${i} ./loop_sigma.out $dir $n_pop $N $K
    echo N${N}K${K}_MEAN_XI_${i} "Running ..."
done
