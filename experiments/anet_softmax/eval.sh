#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd
python setup.py develop


ALL_SPLITS="0 1"
EXP_TAG="softmax"
OOD_SCORING="confidence"


MODEL_OUTPUT=output/anet/${EXP_TAG}/split_{id:d}
ANNO_PATH=datasets/activitynet/annotations_open/split_{id:d}

echo -e "\nClosed Set Evaluation (150 Classes)"
python AFSD/anet/eval_open.py \
    ${MODEL_OUTPUT}/anet_open_rgb.json \
    ${ANNO_PATH}/known_val_gt.json \
    --cls_idx_known ${ANNO_PATH}/action_known.txt \
    --all_splits ${ALL_SPLITS} \
    --ood_scoring ${OOD_SCORING}

echo -e "\nOpen Set Evaluation (150+1 Classes)"
python AFSD/anet/eval_open.py \
    ${MODEL_OUTPUT}/anet_open_rgb.json \
    ${ANNO_PATH}/all_val_gt.json \
    --cls_idx_known ${ANNO_PATH}/action_known.txt \
    --open_set \
    --all_splits ${ALL_SPLITS} \
    --ood_scoring ${OOD_SCORING}

cd $pwd_dir
echo "Experiments finished!"