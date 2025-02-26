#!/bin/bash

qdel submit_opt*
rm *.db
rm -rf results/*
rm ../models/l*
rm ../models/r*
rm ../models/a*
rm submit_optuna_regression_job.*
rm submit_optuna_recurrence_job.*
rm submit_optuna_mosaic_job.*
rm submit_optuna_medici_job.*
echo ""
echo "     remember to stop the job submitting script"
echo ""