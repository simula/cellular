import os
import json
import random
import argparse

from collections import defaultdict

import torch

from torch.utils.data import DataLoader

import numpy as np
import albumentations as albu

from datetime import datetime
from config.utils import replace_val_in_dict

import data

from data.augment import build_augmentations

from utils import append_id
from utils.logging import log_std

from metrics import call_metric

from models import build_model
from loss import build_criterion

from config import Config

random.seed(0)
np.random.seed(0)

argument_parser = argparse.ArgumentParser(description="")
argument_parser.add_argument("-c", "--config_path", nargs="+", required=True)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def train_model(config_path):

    config = Config.from_json(config_path)

    experiment_path = os.path.join(
        os.path.join(ROOT_DIR, "experiments", config.type),
        os.path.splitext(os.path.basename(config_path))[0],
        datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

    config.experiment_path = experiment_path

    replace_val_in_dict("<experiment_path>", experiment_path, config)

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    config.save(os.path.join(experiment_path, "config.json"))

    model_best_save_path = os.path.join(experiment_path, "best_epoch_model.pt")
    model_save_path = os.path.join(experiment_path, "last_epoch_model.pt")

    eval_logs_path = os.path.join(experiment_path, "train_val_log.json")

    L = os.path.join(experiment_path, "history.pkl")
    tensorboard_path = os.path.join(experiment_path, "logs")

    stderr_log = log_std(os.path.join(experiment_path, "stderr.txt"), "stderr")
    stdout_log = log_std(os.path.join(experiment_path, "stdout.txt"), "stdout")

    config.model.params.classes = len(config.train.dataset.classes)

    model = build_model(config.model.name, **config.model.params)

    preprocessing = albu.Compose(
        [albu.Lambda(image=to_tensor, mask=to_tensor)])

    train_dataset_cls = getattr(data, config.train.dataset.name)
    valid_dataset_cls = getattr(data, config.valid.dataset.name)

    train_augmentations = build_augmentations(
        config.train.dataset.augmentations)
    valid_augmentations = build_augmentations(
        config.valid.dataset.augmentations)

    train_dataset = train_dataset_cls(config.train.dataset.root_dir, classes=config.train.dataset.classes,
                                      augmentation=train_augmentations, preprocessing=preprocessing, **config.train.dataset.params)

    valid_dataset = valid_dataset_cls(config.valid.dataset.root_dir, classes=config.valid.dataset.classes,
                                      augmentation=valid_augmentations, preprocessing=preprocessing, **config.valid.dataset.params)

    train_loader = DataLoader(
        train_dataset, batch_size=config.train.dataset.batch_size, shuffle=True, num_workers=12)
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    optimizer = getattr(torch.optim, config.optimizer.name)(
        model.parameters(), **config.optimizer.params)
    criterion = build_criterion(**config.loss)

    dataloaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    metric_logs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

    model = model.to(DEVICE)

    max_score = 0
    since_best = 0
    best_epoch_idx = 0

    training_is_done = False

    for epoch_idx in range(1, config.epochs + 1):

        print("\nEpoch: %i" % (epoch_idx))

        for phase, dataloader in dataloaders.items():

            running_loss = 0

            if phase == "train":
                model.train()
            else:
                model.eval()

            with torch.set_grad_enabled(phase == "train"):

                for i, (inputs, targets) in enumerate(dataloader):

                    print("%s: %i / %i" %
                          (phase, i + 1, len(dataloader)), end="\r")

                    inputs = inputs.to(DEVICE)
                    targets = targets.to(DEVICE)

                    optimizer.zero_grad()
                    outputs = model(inputs)

                    loss = criterion(outputs, targets)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()

                    for metric_name, metric_params in config.metrics.items():
                        metric_logs[phase][epoch_idx][metric_name] += np.mean(call_metric(metric_name, outputs.round().long(), targets.round().long(), metric_params).cpu().numpy(), axis=0)

            for metric_name in config.metrics:
                metric_logs[phase][epoch_idx][metric_name] = metric_logs[phase][epoch_idx][metric_name] / \
                    len(dataloader)

            metric_str = "\t".join(["%s: %.4f" % (metric_name, metric_value.mean().item())
                                   for metric_name, metric_value in metric_logs[phase][epoch_idx].items()])
            watch_metric = metric_logs[phase][epoch_idx][config.optimizer.watch].mean().item()
            print("%s: %i / %i\t %s" %
                  (phase, i + 1, len(dataloader), metric_str))

            if phase == "valid":

                if max_score < watch_metric:
                    since_best = 0
                    best_epoch_idx = epoch_idx
                    print("Best score improved from {:.4f} to {:.4f}".format(
                        max_score, watch_metric))
                    max_score = watch_metric
                    if model_best_save_path is not None:
                        torch.save(model, model_best_save_path)
                        torch.save(model.state_dict(), append_id(
                            model_best_save_path, "state_dict"))
                        torch.save({
                            "epoch": epoch_idx,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()
                        }, append_id(model_best_save_path, "complete"))
                else:
                    since_best += 1

                if config.optimizer.patience is not None and (since_best + 1) % config.optimizer.patience == 0:
                    print("Reducing learning rate...")
                    optimizer.param_groups[0]["lr"] *= 0.1

                if config.patience is not None and since_best >= config.patience:
                    print("Finished training on epoch %i" % epoch_idx)
                    training_is_done = True
                    break

        if training_is_done:
            break

    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError('Not serializable')

    metric_logs["best_epoch"] = best_epoch_idx

    with open(eval_logs_path, "w") as f:
        json.dump(metric_logs, f, indent=4, default=default)

    torch.save(model, model_save_path)

    return model, max_score


if __name__ == "__main__":
    args = argument_parser.parse_args()
    config_paths = args.config_path
    for config_path in config_paths:
        train_model(config_path=config_path)
