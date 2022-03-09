#!/bin/bash

pwd_dir=$pwd
cd ../

source activate opental

GPU_ID=$1
DATASET=thumos14

# run RGB model
echo "Test the ${DATASET} RGB model:"
CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/${DATASET}/test.py \
    configs/${DATASET}.yaml \
    --checkpoint_path=models/${DATASET}/checkpoint-15.ckpt \
    --output_json=${DATASET}_rgb.json

CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/${DATASET}/eval.py output/${DATASET}_rgb.json

# run flow model
echo "Test the ${DATASET} Flow model:"
CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/${DATASET}/test.py \
    configs/${DATASET}_flow.yaml \
    --checkpoint_path=models/${DATASET}_flow/checkpoint-16.ckpt \
    --output_json=${DATASET}_flow.json

CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/${DATASET}/eval.py output/${DATASET}_flow.json

# run fusion (RGB + flow) model
echo "Test the ${DATASET} RGB+Flow model:"
CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/${DATASET}/test.py \
    configs/${DATASET}.yaml \
    --fusion \
    --output_json=${DATASET}_fusion.json

CUDA_VISIBLE_DEVICES=${GPU_ID} python AFSD/${DATASET}/eval.py output/${DATASET}_fusion.json

cd $pwd_dir
echo "Experiments finished!"