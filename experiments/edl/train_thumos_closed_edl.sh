#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd
python setup.py develop

GPU_ID=$1
SPLIT=$2

# run RGB model
echo "Train the RGB model on Thumos14 Closed Set:"
CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/thumos14/train.py \
    configs/thumos14_open_edl_15kc.yaml \
    --lw=1 \
    --cw=10 \
    --ctw=1 \
    --ssl=0.1 \
    --piou=0.5 \
    --open_set \
    --split=${SPLIT}


cd $pwd_dir
echo "Experiments finished!"