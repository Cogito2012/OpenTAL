#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd

GPU_ID=$1
ALL_SPLITS="0 1 2"
OOD_SCORING="confidence"

for SPLIT in ${ALL_SPLITS}
do
    PRED_FILE=output/anet/softmax/split_${SPLIT}/anet_open_trainset.json
    if [ ! -f $PRED_FILE ]; then
        # run RGB model
        echo "Threshold the RGB model on ActivityNet Open Set (Split=${SPLIT}):"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/anet/threshold.py \
            configs/anet_softmax.yaml \
            --open_set \
            --split=${SPLIT} \
            --ood_scoring ${OOD_SCORING} \
            --output_json=anet_open_trainset.json 
    else
        echo "Result file exists! ${PRED_FILE}"
    fi
done


cd $pwd_dir
echo "Experiments finished!"