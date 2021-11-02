#!/bin/bash

source activate afsd

cd ../
python AFSD/anet_data/video2npy.py 8 --video_dir datasets/activitynet/train_val_112 --output_dir datasets/activitynet/train_val_npy_112
cd ./datasets

echo "Done!"
