#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate opental

GPU_ID=$1
ALL_SPLITS="0"
OOD_SCORING="uncertainty"
EXP_TAG="opental_final"

# Thresholding on training set
for SPLIT in ${ALL_SPLITS}
do
    PRED_FILE=output/${EXP_TAG}/split_${SPLIT}/thumos14_open_trainset.json
    if [ ! -f $PRED_FILE ]; then
        # run RGB model
        echo "Threshold the RGB model on Thumos14 Open Set (Split=${SPLIT}):"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/thumos14/threshold.py \
            configs/thumos14_opental_final.yaml \
            --open_set \
            --split=${SPLIT} \
            --ood_scoring ${OOD_SCORING} \
            --output_json=thumos14_open_trainset.json 
    else
        echo "Result file exists! ${PRED_FILE}"
    fi
done


cd $pwd_dir
echo "Experiments finished!"