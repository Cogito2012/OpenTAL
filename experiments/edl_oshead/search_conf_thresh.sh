#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd

GPU_ID=$1
ALL_SPLITS="0"
OOD_SCORING="uncertainty"

PRED_FILE=output/edl_oshead_ghm/split_0/raw_outputs.json
if [ ! -f $PRED_FILE ]; then
    # run RGB model
    CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/thumos14/conf_thresh.py \
        configs/thumos14_open_edl_oshead_tuned.yaml \
        --open_set \
        --split=0 \
        --ood_scoring ${OOD_SCORING}
else
    echo "Result file exists! ${PRED_FILE}"
fi


cd $pwd_dir
echo "Experiments finished!"