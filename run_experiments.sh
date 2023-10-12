#!/bin/bash

INPUT_DIR="./data"
DATA_OUTPUT_DIR="./dev-data"
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

for IMAGE_DATA_NAME in "images-color"; do

    python ./experiments/prepare_datasets.py -i $INPUT_DIR -o $DATA_OUTPUT_DIR/$IMAGE_DATA_NAME -d $IMAGE_DATA_NAME;

    for MODEL_TYPE in cyto cyto2; do

        for CLASS_NAME in all; do
            python ./experiments/train.py \
            --train_dir $DATA_OUTPUT_DIR/$IMAGE_DATA_NAME/$CLASS_NAME/train \
            --valid_dir $DATA_OUTPUT_DIR/$IMAGE_DATA_NAME/$CLASS_NAME/valid \
            --model_type $MODEL_TYPE \
            --experiments_dir $EXPERIMENT_OUTPUT_DIR/$IMAGE_DATA_NAME/$CURRENT_DATE/custom/$MODEL_TYPE \
            --experiment_name $CLASS_NAME --force \
            --n_epochs 500 \
            --learning_rate 0.2 \
            --weight_decay 0.00001 \
            --batch_size 8 \
            --use_sgd true \
            --rescale true
            
            for DIR in test valid train; do
                python ./experiments/eval.py \
                --test_dir $DATA_OUTPUT_DIR/$IMAGE_DATA_NAME/$CLASS_NAME/$DIR \
                --model_path $EXPERIMENT_OUTPUT_DIR/$IMAGE_DATA_NAME/$CURRENT_DATE/$CLASS_NAME/custom/$MODEL_TYPE/models/$CLASS_NAME \
                --output_dir $EXPERIMENT_OUTPUT_DIR/$IMAGE_DATA_NAME/$CURRENT_DATE/$CLASS_NAME/custom/$MODEL_TYPE;
            done
        done

        for CLASS_NAME in all; do
            for DIR in test valid train; do
                python ./experiments/eval_pretrained.py \
                --test_dir $DATA_OUTPUT_DIR/$IMAGE_DATA_NAME/$CLASS_NAME/$DIR \
                --model_type $MODEL_TYPE \
                --output_dir $EXPERIMENT_OUTPUT_DIR/$IMAGE_DATA_NAME/$CURRENT_DATE/$CLASS_NAME/pretrained/$MODEL_TYPE;
            done
        done

    done    
done