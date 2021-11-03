#!/bin/bash

pwd_dir=$pwd
cd ../

source activate afsd

GPU_ID=$1

CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/analyze_gradnorm.py \
    configs/thumos14_open_iou_edl_oshead_ibm.yaml \
    --open_set \
    --split 0