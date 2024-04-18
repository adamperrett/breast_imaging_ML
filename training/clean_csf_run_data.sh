#!/bin/bash

qdel submit_opt*
rm submit_optuna_regression_job.*
rm *.db
rm -rf results/*
rm ../models/l_*
rm ../models/r_*
echo "remember to stop the job submitting script"
