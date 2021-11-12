#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd
python setup.py develop

GPU_ID=$1


CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/analyze_actionness.py \
    configs/thumos14_opental_final.yaml \
    --open_set \
    --split=0

cd $pwd_dir
echo "Experiments finished!"