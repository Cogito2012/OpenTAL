#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate opental

GPU_ID=$1
ALL_SPLITS="0 1 2"
EXP_TAG='openmax'


MODEL_OUTPUT=output/${EXP_TAG}/split_{id:d}/thumos14_open_rgb.json
CLS_IDX_KNOWN=datasets/thumos14/annotations_open/split_{id:d}/Class_Index_Known.txt

echo -e "\nClosed Set Evaluation (15 Classes)"
python AFSD/thumos14/eval_open.py \
    ${MODEL_OUTPUT} \
    datasets/thumos14/annotations_open/split_{id:d}/known_gt.json \
    --cls_idx_known ${CLS_IDX_KNOWN} \
    --all_splits ${ALL_SPLITS}

echo -e "\nOpen Set Evaluation (15+1 Classes)"
python AFSD/thumos14/eval_open.py \
    ${MODEL_OUTPUT} \
    datasets/thumos14/annotations/thumos_gt.json \
    --cls_idx_known ${CLS_IDX_KNOWN} \
    --open_set \
    --draw_auc \
    --all_splits ${ALL_SPLITS}

cd $pwd_dir
echo "Experiments finished!"