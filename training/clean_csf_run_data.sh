#!/bin/bash

qdel submit_opt*
rm submit_optuna_regression_job.*
rm *.db
rm -rf results/*
echo "remember to stop the job submitting script"
