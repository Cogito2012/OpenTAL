#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd
python setup.py develop

GPU_ID=$1
ALL_SPLITS="0 1"
EXP_TAG="edl"
OOD_SCORING="uncertainty"

for SPLIT in ${ALL_SPLITS}
do
    PRED_FILE=output/anet/${EXP_TAG}/split_${SPLIT}/anet_open_rgb.json
    if [ ! -f $PRED_FILE ]; then
        # run RGB model
        echo "Test the RGB model on ActivityNet Open Set (Split=${SPLIT}):"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/anet/test.py \
            configs/anet_edl.yaml \
            --open_set \
            --split=${SPLIT} \
            --output_json=anet_open_rgb.json 
    else
        echo "Result file exists! ${PRED_FILE}"
    fi
done

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
    --trainset_result ${MODEL_OUTPUT}/anet_open_trainset.json \
    --all_splits ${ALL_SPLITS} \
    --ood_scoring ${OOD_SCORING}

cd $pwd_dir
echo "Experiments finished!"