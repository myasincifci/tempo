#!/usr/bin/env bash
#$ -binding linear:4 # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)
#$ -N Baseline       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)
#$ -cwd        # change working directory (to current)
#$ -V          # provide environment variables
#$ -t 1-5      # start 100 instances: from 1 to 100
#$ -l cuda=1   # request one GPU

array=("x" "1" "5" "10" "15" "20")
../env/bin/python tools/semi_sup_eval.py \
    --path model_zoo/baseline.pth \
    --runs 100 \
    --name semi_sup_eval/baseline \
    --samples_pc "${array[$SGE_TASK_ID]}"