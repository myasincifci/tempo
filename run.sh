#!/usr/bin/env bash
#$ -binding linear:4 # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)
#$ -N tempo       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)
#$ -cwd        # change working directory (to current)
#$ -V          # provide environment variables
#$ -t 1-5      # start 100 instances: from 1 to 100
#$ -l cuda=1   # request one GPU

../env/bin/python tools/train_net_new.py --basline True --eval finetune