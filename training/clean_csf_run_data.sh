#!/bin/bash

qdel submit_opt*
rm *.db
rm -rf results/*
rm ../models/l_*
rm ../models/r_*
rm submit_optuna_regression_job.*
echo ""
echo "remember to stop the job submitting script"
echo ""