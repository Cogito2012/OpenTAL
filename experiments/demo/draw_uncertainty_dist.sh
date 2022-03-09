#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate opental
python setup.py develop

GPU_ID=$1
OOD_SCORING="uncertainty_actionness"


CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/thumos14/draw_distribution.py \
    configs/thumos14_open_iou_edl_oshead_15kc.yaml \
    --open_set \
    --split=0 \
    --ood_scoring ${OOD_SCORING}

cd $pwd_dir
echo "Experiments finished!"