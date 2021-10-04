#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd

GPU_ID=$1
ALL_SPLITS="0 1 2 4"

for SPLIT in ${ALL_SPLITS}
do
    PRED_FILE=output/efl_15kc/split_${SPLIT}/thumos14_open_trainset.json
    if [ ! -f $PRED_FILE ]; then
        # run RGB model
        echo "Threshold the RGB model on Thumos14 Open Set (Split=${SPLIT}):"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/thumos14/threshold.py \
            configs/thumos14_open_efl_15kc.yaml \
            --open_set \
            --split=${SPLIT} \
            --ood_scoring uncertainty \
            --output_json=thumos14_open_trainset.json 
    else
        echo "Result file exists! ${PRED_FILE}"
    fi
done


cd $pwd_dir
echo "Experiments finished!"