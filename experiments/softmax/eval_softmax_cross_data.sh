#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate opental
python setup.py develop

ALL_SPLITS="0 1 2"
EXP_TAG="softmax_crossdata"
OOD_SCORING="confidence"


MODEL_OUTPUT=output/${EXP_TAG}/split_{id:d}/thumos14_anet_merged.json
CLS_IDX_KNOWN=datasets/thumos14/annotations_open/split_{id:d}/Class_Index_Known.txt


echo -e "\nOpen Set Cross-Data Evaluation (15+1 Classes)"
python AFSD/thumos14/eval_open.py \
    ${MODEL_OUTPUT} \
    datasets/thumos14/annotations/thumos_anet_gt.json \
    --cls_idx_known ${CLS_IDX_KNOWN} \
    --open_set \
    --dataset thumos_anet \
    --all_splits ${ALL_SPLITS} \
    --ood_scoring ${OOD_SCORING}

cd $pwd_dir
echo "Experiments finished!"