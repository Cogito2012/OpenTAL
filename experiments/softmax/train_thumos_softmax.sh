#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd

GPU_ID=$1
SPLIT=$2

# run RGB model
echo "Train the RGB model on Thumos14 Closed Set:"
CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/thumos14/train.py \
    configs/thumos14_softmax.yaml \
    --lw=10 \
    --cw=1 \
    --piou=0.5 \
    --open_set \
    --split=${SPLIT}


cd $pwd_dir
echo "Experiments finished!"