#!/bin/bash

pwd_dir=$pwd
cd ../../

source activate afsd
python setup.py develop

GPU_ID=$1
ALL_SPLITS="0 1"
EXP_TAG="softmax"

for SPLIT in ${ALL_SPLITS}
do
    PRED_FILE=output/anet/${EXP_TAG}/split_${SPLIT}/anet_open_rgb.json
    if [ ! -f $PRED_FILE ]; then
        # run RGB model
        echo "Test the RGB model on ActivityNet Open Set (Split=${SPLIT}):"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/anet/test.py \
            configs/anet_softmax.yaml \
            --open_set \
            --split=${SPLIT} \
            --output_json=anet_open_rgb.json 
    else
        echo "Result file exists! ${PRED_FILE}"
    fi
done

cd $pwd_dir
echo "Experiments finished!"