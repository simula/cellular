import os
import random
import argparse
import torch
import cv2
import json

import numpy as np

import numpy as np
import albumentations as albu

from torch.utils.data import DataLoader

import data
from data.augment import build_augmentations
from models import build_model

from config import Config

import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

# from sklearn.metrics import jaccard_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import accuracy_score

random.seed(0)
np.random.seed(0)

argument_parser = argparse.ArgumentParser(description="")
argument_parser.add_argument("-c", "--config_path", required=True, nargs="+")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(DEVICE)

_COLOR_MAPPING = {
    0: [0, 0, 0], # black
    1: [255, 0, 0], # red
    2: [0, 255, 0], # green
    3: [0, 0, 255], # blue
    4: [255, 255, 0], # yellow
    5: [0, 255, 255] # teal
}

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def display_multiple_img(images, rows=1, cols=1, output_path=None):
    plt.cla()
    plt.clf()
    plt.close()
    figure, ax = plt.subplots(nrows=rows, ncols=cols, gridspec_kw={"wspace":1,"hspace":1}, figsize=(15,10))

    for y in range(rows):
        for x in range(cols):
            ax[ y, x ].axis("off")
            ax[ y, x ].set_title(images[ y ][ x ][0])
            ax[ y, x ].imshow(images[ y ][ x ][1])
            ax[ y, x ].set_axis_off()

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()

def merge_masks(masks):
    output = np.zeros((*masks.shape[1:], 3))
    for index, mask in enumerate(masks):
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] > 0:
                    output[y, x] = _COLOR_MAPPING[index]
    return output

def dice_score(y_true, y_pred):
    return np.sum(y_pred[y_true == 1] == 1) * 2.0 / (np.sum(y_pred[y_pred == 1] == 1) + np.sum(y_true[y_true == 1] == 1))

def test_model(config_path):

    config = Config.from_json(config_path)

    model = build_model(config.model.name, **config.model.params)
    model = torch.load(os.path.join(config.experiment_path, "best_epoch_model.pt"))

    preprocessing = albu.Compose([ albu.Lambda(image=to_tensor, mask=to_tensor) ])

    test_dataset_cls = getattr(data, config.test.dataset.name)
    test_augmentations = build_augmentations(config.test.dataset.augmentations)

    test_dataset = test_dataset_cls(config.test.dataset.root_dir, classes=config.test.dataset.classes,
        augmentation=test_augmentations, preprocessing=preprocessing, **config.test.dataset.params)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    if config.optimizer.watch not in config.metrics and config.optimizer.watch is not None:
        config.metrics.append(config.optimizer.watch)

    model = model.to(DEVICE)

    mean_score = []

    for i, (inputs, targets) in enumerate(test_loader):

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)

        y_pred = outputs.detach().cpu().round().long()
        y_true = targets.detach().cpu().round().long()

        tp, fp, fn, tn = smp.metrics.get_stats(y_pred, y_true, mode='multilabel', threshold=0.5)

        # old_score_f1 = f_score(y_pred, y_true, threshold=.5)
        # old_score_jaccard = iou(y_pred, y_true, threshold=.5)
        # old_score_recall = recall(y_pred, y_true, threshold=.5)
        # old_score_precision = precision(y_pred, y_true, threshold=.5)

        score_jaccard = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
        score_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)
        score_recall = smp.metrics.recall(tp, fp, fn, tn, reduction=None)
        score_precision = smp.metrics.precision(tp, fp, fn, tn, reduction=None)

        print(score_jaccard, score_f1, score_recall, score_precision)
        # print(old_score_jaccard, old_score_f1, old_score_recall, old_score_precision)

        mean_score.append([
            score_jaccard,
            score_f1,
            score_recall,
            score_precision,
        ])

        mean_score.append([
            score_jaccard,
            score_f1,
            score_recall,
            score_precision,
        ])

    mean_score = np.mean(mean_score, axis=0)

    test_logs = {
        "score_jaccard": str(mean_score[0]),
        "score_f1": str(mean_score[1]),
        "score_recall": str(mean_score[2]),
        "score_precision": str(mean_score[3]),
    }

    with open(os.path.join(config.experiment_path, "test_evaluation.json"), "w") as f:
        json.dump(test_logs, f, indent=4)

    eval_true_dir = os.path.join(config.experiment_path, "eval", "true")
    eval_pred_dir = os.path.join(config.experiment_path, "eval", "pred")

    if not os.path.exists(eval_true_dir):
        os.makedirs(eval_true_dir)
    
    if not os.path.exists(eval_pred_dir):
        os.makedirs(eval_pred_dir)

    for i, (x_item, y_true) in enumerate(test_loader):

        image_path = test_loader.dataset.image_paths[ i ]
        image_name = os.path.basename(image_path)
        image_save_name = image_name.split(".")[0] + ".png"

        image = cv2.imread(image_path, 0)
        image_shape = image.shape[:2]

        print("Processing %s..." % image_name)

        x_item = x_item.to(DEVICE)
        y_true = y_true.to(DEVICE)

        with torch.set_grad_enabled(False):
            y_pred = model(x_item)

        x_item = x_item.detach().cpu().numpy().squeeze()
        x_item = np.moveaxis(x_item, 0, 2)

        y_true = y_true.detach().cpu().numpy().squeeze()
        y_pred = y_pred.detach().cpu().numpy().squeeze().round()

        y_true = cv2.resize(merge_masks(y_true), image_shape)
        y_pred = cv2.resize(merge_masks(y_pred), image_shape)

        cv2.imwrite(os.path.join(eval_true_dir, image_save_name), y_true)
        cv2.imwrite(os.path.join(eval_pred_dir, image_save_name), y_pred)

if __name__ == "__main__":
    args = argument_parser.parse_args()
    config_paths = args.config_path
    for config_path in config_paths:
        test_model(config_path=config_path)