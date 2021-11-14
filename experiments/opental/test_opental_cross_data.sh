#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd

GPU_ID=$1
ALL_SPLITS="0 1 2"
EXP_TAG="opental_crossdata"


# Inference on testing set
for SPLIT in ${ALL_SPLITS}
do
    PRED_FILE=output/${EXP_TAG}/split_${SPLIT}/thumos14_anet_merged.json
    if [ ! -f $PRED_FILE ]; then
        # run RGB model
        echo "Test the RGB model on Thumos14+ActivityNet1.3 Open Set (Thumos Split=${SPLIT}):"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/thumos14/test_cross_data.py \
            configs/thumos14_opental_final.yaml \
            --open_set \
            --split=${SPLIT} \
            --output_json=thumos14_anet_merged.json 
    else
        echo "Result file exists! ${PRED_FILE}"
    fi
done


cd $pwd_dir
echo "Experiments finished!"