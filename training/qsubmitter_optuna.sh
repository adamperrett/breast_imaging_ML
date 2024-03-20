#!/bin/bash

# Desired number of parallel jobs
MAX_JOBS=6
# Total number of trials you want to run
TOTAL_TRIALS=1000
# Trials counter
TRIALS_COUNTER=0

while [ $TRIALS_COUNTER -lt $TOTAL_TRIALS ]
do
  # Count the number of running jobs (adjust the pattern to match your job names)
  RUNNING_JOBS=$(qstat | grep -c 'submit_CSF')

  # Check if we can submit more jobs
  if [ $RUNNING_JOBS -lt $MAX_JOBS ]
  then
    # Submit a new job
    ./qsub_regression_shell.sh
    let TRIALS_COUNTER=TRIALS_COUNTER+1
    echo "Submitted trial $TRIALS_COUNTER at $(date '+%Y-%m-%d %H:%M:%S')"
  else
    # Wait a bit before checking again
    sleep 1m
  fi
done
