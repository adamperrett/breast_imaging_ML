#!/bin/bash

scancel -u $USER
rm *.db
rm -rf results/*
rm ../models/l_*
rm ../models/r_*
rm ../models/a_*
rm submit_optuna_regression_job.*
rm submit_optuna_recurrence_job.*
rm submit_optuna_mosaic_job.*
rm submit_optuna_medici_job.*
rm submit_optuna_CRUK_job.*
rm slurm-*
echo ""
echo "     remember to stop the job submitting script"
echo ""