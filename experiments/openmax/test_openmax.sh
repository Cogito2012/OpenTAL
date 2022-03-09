#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate opental

GPU_ID=$1
ALL_SPLITS="0 1 2"
EXP_TAG='openmax'

# for demo only
for SPLIT in ${ALL_SPLITS}
do
    PRED_FILE=output/${EXP_TAG}/split_${SPLIT}/thumos14_open_rgb.json
    if [ ! -f $PRED_FILE ]; then
        # run RGB model
        echo "Test the RGB model on Thumos14 Open Set (Split=${SPLIT}):"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/thumos14/test_openmax.py \
            configs/thumos14_openmax.yaml \
            --open_set \
            --split=${SPLIT} \
            --output_json=thumos14_open_rgb.json 
    else
        echo "Result file exists! ${PRED_FILE}"
    fi
done

cd $pwd_dir
echo "Experiments finished!"