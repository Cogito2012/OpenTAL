#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate opental
python setup.py develop

ALL_SPLITS="0 1 2"
EXP_TAG="opental_final"
OOD_SCORING="uncertainty"
# OOD_SCORING="confidence"  # s=1-max(P)
# OOD_SCORING="uncertainty_actionness"  # s=u*a
# OOD_SCORING="a_by_inv_u"  # s=a/(1-u)
# OOD_SCORING="u_by_inv_a"  # s=u/(1-a)


MODEL_OUTPUT=output/${EXP_TAG}/split_{id:d}/thumos14_open_rgb.json
CLS_IDX_KNOWN=datasets/thumos14/annotations_open/split_{id:d}/Class_Index_Known.txt

echo -e "\nClosed Set Evaluation (15 Classes)"
python AFSD/thumos14/eval_open.py \
    ${MODEL_OUTPUT} \
    datasets/thumos14/annotations_open/split_{id:d}/known_gt.json \
    --cls_idx_known ${CLS_IDX_KNOWN} \
    --all_splits ${ALL_SPLITS} \
    --ood_scoring ${OOD_SCORING}

echo -e "\nOpen Set Evaluation (15+1 Classes)"
python AFSD/thumos14/eval_open.py \
    ${MODEL_OUTPUT} \
    datasets/thumos14/annotations/thumos_gt.json \
    --cls_idx_known ${CLS_IDX_KNOWN} \
    --open_set \
    --draw_auc \
    --all_splits ${ALL_SPLITS} \
    --ood_scoring ${OOD_SCORING}

cd $pwd_dir
echo "Experiments finished!"