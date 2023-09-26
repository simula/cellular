import argparse
import os
import json
from cellpose import io, models
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import colorsys
from PIL import Image
import random
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score


def generate_colors(num_classes):
    """
    Generates a list of distinct, contrasting RGB colors for the given number of classes.

    Parameters:
    - num_classes: The number of distinct colors to generate.

    Returns:
    - List of RGB colors.
    """
    colors = [(0, 0, 0)]  # Set the first color as black for the background
    for i in range(
        1, num_classes
    ):  # Start from 1 to avoid overwriting the background color
        hue = i / float(random.random())
        rgb = tuple(round(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
        colors.append(rgb)
    return colors


def visualize_mask_with_pil(mask, save_path):
    """
    Visualizes a semantic segmentation mask using PIL with automatically generated colors.

    Parameters:
    - mask: A 2D numpy array where each value represents a class label.

    Returns:
    - None
    """
    unique_classes = np.unique(mask)
    num_classes = len(unique_classes)
    colors = generate_colors(num_classes)

    class_to_color = {unique_classes[i]: colors[i] for i in range(num_classes)}

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_label, color in class_to_color.items():
        color_mask[mask == class_label] = color

    img = Image.fromarray(color_mask, "RGB")
    img.save(save_path)


def main(args):
    logger, _ = io.logger_setup()

    channels = [0, 0]

    jaccard_list, precision_list, recall_list, f1_list = [], [], [], []

    test_dir = args.test_dir
    model_path = args.model_path
    output_dir = args.output_dir

    dataset_name = os.path.basename(test_dir)

    image_output_dir = os.path.join(output_dir, "images")

    results = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)

    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    logger.info("Loading test data...")
    output = io.load_images_labels(test_dir)
    test_data, test_labels, test_filenames = output

    for inputs, labels, abs_filename in zip(test_data, test_labels, test_filenames):
        filename = os.path.basename(abs_filename)

        masks, _, _ = model.eval(inputs, channels=channels, diameter=None)

        fig, ax = plt.subplots()

        colors = [(0, 0, 0)] + [(plt.cm.tab10(i / (1000 - 1))) for i in range(1, 1000)]
        cmap = matplotlib.colors.ListedColormap(colors)

        ax.axis("off")
        ax.imshow(masks.astype(np.uint8), cmap=cmap)
        fig.savefig(
            os.path.join(image_output_dir, "pred_visualized_%s" % filename),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

        visualize_mask_with_pil(
            masks, os.path.join(image_output_dir, "pred_PIL_%s" % filename)
        )

        io.imsave(os.path.join(image_output_dir, "pred_%s" % filename), masks)

        masks = masks.flatten()
        masks[masks > 0] = 1

        labels = labels.flatten()
        labels[labels > 0] = 1

        precision = precision_score(labels, masks, average=None)
        recall = recall_score(labels, masks, average=None)
        f1 = f1_score(labels, masks, average=None)
        jaccard = jaccard_score(labels, masks, average=None)

        logger.info(f"Filename: {filename}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 Score: {f1}")
        logger.info(f"Jaccard: {jaccard}")

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        jaccard_list.append(jaccard)

    precision_list = np.array(precision_list)
    recall_list = np.array(recall_list)
    f1_list = np.array(f1_list)
    jaccard_list = np.array(jaccard_list)

    results["background"] = {
        "precision": np.mean(precision_list[:, 0]),
        "recall": np.mean(recall_list[:, 0]),
        "f1": np.mean(f1_list[:, 0]),
        "jaccard": np.mean(jaccard_list[:, 0]),
    }

    results["cell"] = {
        "precision": np.mean(precision_list[:, 1]),
        "recall": np.mean(recall_list[:, 1]),
        "f1": np.mean(f1_list[:, 1]),
        "jaccard": np.mean(jaccard_list[:, 1]),
    }

    results["average"] = {
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1": np.mean(f1_list),
        "jaccard": np.mean(jaccard_list),
    }

    with open(os.path.join(output_dir, "%s_eval.json" % dataset_name), "w") as f:
        json.dump(results, f, indent=4)
        logger.info("Evaluation results saved to eval.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset paths and model name argument parser"
    )

    parser.add_argument("--test_dir", type=str, help="Path to the testing directory")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument(
        "--output_dir", type=str, help="Directory to output images and evaluation"
    )

    args = parser.parse_args()
    main(args)
