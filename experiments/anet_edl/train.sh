#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd
python setup.py develop

GPU_ID=$1
SPLIT=$2

# run RGB model
echo "Train the RGB model on ActivityNet1.3 Closed Set:"
CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/anet/train.py \
    configs/anet_edl.yaml \
    --lw=1 \
    --cw=1 \
    --piou=0.6 \
    --open_set \
    --split=${SPLIT}


cd $pwd_dir
echo "Experiments finished!"