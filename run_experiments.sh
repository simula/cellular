#!/bin/bash

INPUT_DIR=${1:-"default/input/path"}
DATA_OUTPUT_DIR=${2:-"default/output/path"}
EXPERIMENT_OUTPUT_DIR=${3:-"default/experiment/path"}

python ./experiments/prepare_date.py -i $INPUT_DIR -o $DATA_OUTPUT_DIR

for CLASS_NAME in all fed unfed; do
    python ./experiments/train.py \
    --train_dir $DATA_OUTPUT_DIR/$CLASS_NAME/train \
    --valid_dir $DATA_OUTPUT_DIR/$CLASS_NAME/valid \
    --experiments_dir $EXPERIMENT_OUTPUT_DIR \
    --experiment_name $CLASS_NAME
    
    for DIR in test valid train; do
        python ./experiments/eval.py \
        --test_dir $DATA_OUTPUT_DIR/$CLASS_NAME/$DIR \
        --model_path $EXPERIMENT_OUTPUT_DIR/$CLASS_NAME/models/$CLASS_NAME \
        --output_dir $EXPERIMENT_OUTPUT_DIR/$CLASS_NAME
    done
done
