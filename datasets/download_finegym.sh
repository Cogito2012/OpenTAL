#!/usr/bin/env bash

# set up environment
source activate afsd
pip install --upgrade youtube-dl mmcv

set -e

DATA_DIR="./finegym"
ANNO_DIR="./finegym/annotations"

if [[ ! -d "${ANNO_DIR}" ]]; then
  echo "${ANNO_DIR} does not exist. Creating";
  mkdir -p ${ANNO_DIR}
  # downloading the annotation files
  echo "Downloading annotations..."
  wget https://sdolivia.github.io/FineGym/resources/dataset/finegym_annotation_info_v1.0.json -O $ANNO_DIR/annotation.json
  wget https://sdolivia.github.io/FineGym/resources/dataset/gym99_train_element_v1.0.txt -O $ANNO_DIR/gym99_train_org.txt
  wget https://sdolivia.github.io/FineGym/resources/dataset/gym99_val_element.txt -O $ANNO_DIR/gym99_val_org.txt
fi

echo "Downloading videos..."
python download.py ${ANNO_DIR}/annotation.json ${DATA_DIR}/videos