#!/bin/bash

INPUT_DIR="./data"
DATA_OUTPUT_DIR="./data/dev-data"
EXPERIMENT_OUTPUT_DIR="./results"
CURRENT_DATE=$(date +"%D/%T")

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input-dir) INPUT_DIR="$2"; shift ;;
        -o|--data-output-dir) DATA_OUTPUT_DIR="$2"; shift ;;
        -e|--experiment-output-dir) EXPERIMENT_OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python ./experiments/prepare_datasets.py -i $INPUT_DIR -o $DATA_OUTPUT_DIR

for CLASS_NAME in all fed unfed; do
    python ./experiments/train.py \
    --train_dir $DATA_OUTPUT_DIR/$CLASS_NAME/train \
    --valid_dir $DATA_OUTPUT_DIR/$CLASS_NAME/valid \
    --experiments_dir $EXPERIMENT_OUTPUT_DIR/$CURRENT_DATE \
    --experiment_name $CLASS_NAME --force
    
    for DIR in test valid train; do
        python ./experiments/eval.py \
        --test_dir $DATA_OUTPUT_DIR/$CLASS_NAME/$DIR \
        --model_path $EXPERIMENT_OUTPUT_DIR/$CURRENT_DATE/$CLASS_NAME/models/$CLASS_NAME \
        --output_dir $EXPERIMENT_OUTPUT_DIR/$CURRENT_DATE/$CLASS_NAME
    done
done