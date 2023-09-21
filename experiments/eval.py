import argparse
import os
import json
from cellpose import io, models
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

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

        masks, _, _ = model.eval(
            inputs,
            channels=channels
        )

        fig, ax = plt.subplots()

        colors = [(0, 0, 0)] + [(plt.cm.tab10(i/(1000-1))) for i in range(1, 1000)]
        cmap = matplotlib.colors.ListedColormap(colors)

        ax.imshow(masks, cmap=cmap)
        ax.axis('off')
        fig.savefig(os.path.join(image_output_dir, "pred_visualized_%s" % filename), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset paths and model name argument parser')

    parser.add_argument('--test_dir', type=str, help='Path to the testing directory')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--output_dir', type=str, help='Directory to output images and evaluation')

    args = parser.parse_args()
    main(args)