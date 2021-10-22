#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd
python setup.py develop

GPU_ID=$1
OOD_SCORING="uncertainty"


CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/thumos14/draw_distribution.py \
    configs/thumos14_open_edl_15kc.yaml \
    --open_set \
    --split=0 \
    --ood_scoring ${OOD_SCORING}

cd $pwd_dir
echo "Experiments finished!"