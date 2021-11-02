#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd

GPU_ID=$1
ALL_SPLITS="0 1 2"
EXP_TAG="edl_15kc"

for SPLIT in ${ALL_SPLITS}
do
    PRED_FILE=output/${EXP_TAG}/split_${SPLIT}/thumos14_open_rgb.json
    if [ ! -f $PRED_FILE ]; then
        # run RGB model
        echo "Test the RGB model on Thumos14 Open Set (Split=${SPLIT}):"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/thumos14/test.py \
            configs/thumos14_open_edl_15kc.yaml \
            --open_set \
            --split=${SPLIT} \
            --output_json=thumos14_open_rgb.json 
    else
        echo "Result file exists! ${PRED_FILE}"
    fi
done

MODEL_OUTPUT=output/${EXP_TAG}/split_{id:d}/thumos14_open_rgb.json
CLS_IDX_KNOWN=datasets/thumos14/annotations_open/split_{id:d}/Class_Index_Known.txt
TRAINSET_RESULT=output/${EXP_TAG}/split_{id:d}/thumos14_open_trainset.json

echo -e "\nClosed Set Evaluation (15 Classes)"
python AFSD/thumos14/eval_open.py \
    ${MODEL_OUTPUT} \
    datasets/thumos14/annotations_open/split_{id:d}/known_gt.json \
    --cls_idx_known ${CLS_IDX_KNOWN} \
    --all_splits ${ALL_SPLITS} \
    --ood_scoring uncertainty

echo -e "\nOpen Set Evaluation (15+1 Classes)"
python AFSD/thumos14/eval_open.py \
    ${MODEL_OUTPUT} \
    datasets/thumos14/annotations/thumos_gt.json \
    --cls_idx_known ${CLS_IDX_KNOWN} \
    --open_set \
    --draw_auc \
    --trainset_result ${TRAINSET_RESULT} \
    --all_splits ${ALL_SPLITS} \
    --ood_scoring uncertainty

cd $pwd_dir
echo "Experiments finished!"