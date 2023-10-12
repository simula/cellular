# CELLULAR

![banner](/static/images/banner.png)

Life is in an equilibrium state between synthesis and degradation of proteins, and that is the difference between organisms and machines! Yoshinori Ohsumi. Ohsumi was awarded the 2016 Nobel Prize in Physiology or Medicine for his discoveries of mechanisms for autophagy. “Autophagy” originates from the Greek words auto-, meaning “self”, and phagein, meaning “to eat”, and hence denotes “self-eating”. Autophagy involves pathways undertaken by animal cells to deliver cytoplasmic material to the lysosome for degradation as means of cellular regeneration. A series of molecular events culminate in the formation of the autophagosomes, their subsequent fusion with lysosomes and finally degradation of cargo in autolysosomes.

Autophagy is a dynamic process that exists in basal and activated levels. Basal autophagy, defined as autophagic activity occurring during cellular growth in nutrient-rich media, ensures maintenance of cellular quality control such as regular recycling of unwanted or damaged cellular products. On the other hand, stimulated levels exist in response to stressful conditions, such as changes in nutritional status and the presence of abnormal proteins as means to ensure protection from stress-induced damage. The hallmark of autophagy is the formation of double-membrane vesicles known as autophagosomes, which engulf cytoplasmic material and deliver the cargo to hydrolytic enzymes in lysosomes for degradation. Hence, autophagy provides building blocks for synthesis of macromolecules during limitations in nutrient supply.

The dataet is available here: https://zenodo.org/record/8315423

## Reproducibility Guide
The following steps can be used to reproduce the experiments included in the paper.

The weights for the models presented in the paper are available here:
* [Fine-tuned cyto](https://datasets.simula.no/downloads/cellular-experiments-models/Fine-Tuned-Cyto)
* [Fine-tuned cyto2](https://datasets.simula.no/downloads/cellular-experiments-models/Fine-Tuned-Cyto2)

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- OpenCV (cv2)
- NumPy
- Cellpose
- Matplotlib
- Scikit-Learn

You can install the necessary Python packages by running:

```bash
pip install -r requirements.txt
```

You can reproduce all experiemnts as they were presented in the paper by running the `run_experiments.sh` shell script like the following: 
```bash
sh run_experiments.sh ./path/to/masks ./output-path
```

### Stage 1: Prepare Data
The `prepare_data.py` script contains functions to process and combine mask images. Here is a brief description of each function:

- `setup_logging`: Sets up logging configuration.
- `load_mask_images`: Loads grayscale images from a specified folder.
- `copy_corresponding_files`: Copies files corresponding to the mask based on the filename.
- `combine_masks`: Combines multiple mask images into a single image with distinct object IDs.
- `create_directory`: Creates a directory if it does not exist.
- `split_data`: Splits data into train, validation, and test sets based on specific string patterns.
- `process_masks`: Processes and saves combined mask images.

#### How to Run
To run the script, navigate to the directory containing the `prepare_data.py` script and execute the following command:

```bash
python prepare_data.py -i <input_path> -o <output_dir> -c <corresponding_files_dir> -v <val_pattern> -t <test_pattern> -f <formats>
```

### Stage 2: Train Model

The `train.py` script facilitates the training of the model using the Cellpose library.

Parameters explained:
- `--train_dir`: Path to the directory containing training data.
- `--valid_dir`: Path to the directory containing validation data.
- `--model_type`: Type of model to fine-tune.
- `--experiment_name`: The name of the experiment (used to name the output directory and saved model).
- `--experiments_dir`: The directory where experiment outputs (images, evaluations) will be saved (default is "./experiments").
- `--n_epochs`: The number of training epochs
- `--learning_rate`: The learning rate
- `--weight_decay`: The weight decay
- `--batch_size`: The batch size
- `--use_sgd`: Using SGD or Radam
- `--rescale`: Wether to rescale cells

Specifics on the trainig api can be found in Cellpose's official documentation: https://cellpose.readthedocs.io/en/latest/api.html

#### How to Run
To run the script, navigate to the directory containing the `train_model.py` script and execute the following command:

```bash
python train.py --train_dir <path_to_training_directory> --valid_dir <path_to_validation_directory> --experiment_name <experiment_name> --experiments_dir <path_to_experiments_directory>
```

#### Example Command

```bash
python train.py --train_dir "/path/to/train/data" --valid_dir "/path/to/valid/data" --experiment_name "experiment_1" --experiments_dir "./experiments"
```

This command will train a model using the data in `/path/to/train/data` as the training data and `/path/to/valid/data` as the validation data. The outputs will be saved in the `./experiments` directory under a folder named "experiment_1".

### Stage 3: Evaluate Model

The `eval.py` script is responsible for evaluating the performance of the trained Cellpose model on a test dataset.

Parameters explained:
- `--test_dir`: Path to the directory containing the test data.
- `--model_path`: Path to the trained model file.
- `--output_dir`: The directory where the evaluation results and predicted masks will be saved.

#### How to Run
To run the script, navigate to the directory containing the `evaluate_model.py` script and execute the following command:

```bash
python eval.py --test_dir <path_to_test_directory> --model_path <path_to_pretrained_model> --output_dir <path_to_output_directory>
```

#### Example Command

```bash
python eval.py --test_dir "/path/to/test/data" --model_path "/path/to/model" --output_dir "./experiments/evaluation_results"
```

This command will evaluate the model using the data in `/path/to/test/data` as the test data and `/path/to/model` as the pretrained model. The outputs, including evaluation results and predicted masks, will be saved in the `./experiments/evaluation_results` directory.

## Support
Please contact steven@simula.no or vajira@simula.no for any questions regarding the dataset.

## Terms of Use
The data is released fully open for research and educational purposes. The use of the dataset for purposes such as competitions and commercial purposes needs prior written permission. In all documents and papers that use or refer to the dataset or report experimental results based on the CELLULAR.
