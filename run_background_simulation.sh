#!/bin/bash
logdir=simulation_log
mkdir -p $logdir
time_suf=$(date "+%Y%m%d-%H%M%S")
logfile="$logdir/simulation_$time_suf.log"
python3 paper_simulations.py $1 $2 > $logfile 2>&1 & 
echo "Backgrounded simulation, logfile $logfile."
