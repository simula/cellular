import os
import cv2

import zipfile
import requests
import argparse
import logging

import numpy as np
from shutil import copy2
from collections import defaultdict

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_mask_images(folder_path):
    """Load grayscale images from the specified folder based on the base file name."""
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            logging.warning(f"Failed to read image: {filename}")
    return images

def copy_corresponding_files(corresponding_files_dir, output_dir, filename_base):
    """Copy files that correspond to the mask based on the filename."""
    for file in os.listdir(corresponding_files_dir):
        if file.startswith(filename_base):
            source_file_path = os.path.join(corresponding_files_dir, file)
            destination_file_path = os.path.join(output_dir, file)
            copy2(source_file_path, destination_file_path)
            logging.info(f"Copied corresponding file: {file}")

def combine_masks(mask_images):
    """Combine multiple mask images into a single image with distinct object IDs."""
    height, width = mask_images[0].shape
    combined_mask = np.zeros((height, width), dtype=np.uint16)
    object_id = 1
    
    for mask in mask_images:
        indices = np.where(mask > 0)
        combined_mask[indices] = object_id
        object_id += 1

    return combined_mask

def create_directory(directory_path):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")

def split_data(directory_path, val_pattern, test_pattern):
    """Split data into train, validation, and test sets based on specific string patterns."""
    data = defaultdict(list)

    for dirname in os.listdir(directory_path):
        if val_pattern in dirname:
            data["valid"].append(os.path.join(directory_path, dirname))
        elif test_pattern in dirname:
            data["test"].append(os.path.join(directory_path, dirname))
        else:
            data["train"].append(os.path.join(directory_path, dirname))

    return data

def process_masks(input_path, image_dirname, mask_dirname, output_dir, val_pattern, test_pattern, format, exclude_class=["Unidentified"]):
    """Process and save combined mask images."""
    if not os.path.exists(input_path):
        logging.error(f"The input path {input_path} does not exist.")
        return
    
    masks_dir = os.path.join(input_path, mask_dirname)
    images_dir = os.path.join(input_path, image_dirname)
    
    # Split data into train, validation, and test sets
    data = split_data(masks_dir, val_pattern, test_pattern)
    
    # Process and save combined masks for each split
    for split_name, split_files in data.items():

        for dirpath in split_files:

            filename_base = os.path.basename(dirpath)

            all_masks = []
            all_masks_output_dir = os.path.join(output_dir, "all", split_name)
            create_directory(all_masks_output_dir)
            logging.info(f"Processing {dirpath}...")
            
            for class_name in os.listdir(dirpath):
                
                if class_name in exclude_class:
                    logging.info(f"Skipping {class_name}...")
                    continue

                mask_images = load_mask_images(os.path.join(dirpath, class_name))
                class_masks = combine_masks(mask_images)
                all_masks.extend(class_masks)

                # Create output directory structure: output_dir/class_name/split_name
                output_dir_class_split = os.path.join(output_dir, class_name, split_name)
                create_directory(output_dir_class_split)

                save_filepath = os.path.join(output_dir_class_split, f"{filename_base}_masks.{format}")
                cv2.imwrite(save_filepath, class_masks)
                logging.info(f"Combined mask image saved as {save_filepath}")

                # Copy the corresponding files based on the filename of the mask
                copy_corresponding_files(images_dir, output_dir_class_split, filename_base)

            combined_mask = combine_masks(mask_images)
            save_filepath = os.path.join(all_masks_output_dir, f"{filename_base}_masks.{format}")
            cv2.imwrite(save_filepath, combined_mask)
            logging.info(f"Combined mask image saved as {save_filepath}")

            # Copy the corresponding files based on the filename of the mask
            copy_corresponding_files(images_dir, all_masks_output_dir, filename_base)

def download_and_unzip(url, directory_path):
    # Check if directory exists
    if not os.path.exists(directory_path):

        logging.info("Input dataset does not exist.")
        logging.info("Downloading")

        os.makedirs(directory_path)

        # Download the zip file
        zip_file_path = os.path.join(directory_path, "temp.zip")
        response = requests.get(url)
        
        if response.status_code == 200:

            logging.info("Download complete.")

            with open(zip_file_path, 'wb') as zip_file:
                zip_file.write(response.content)

            logging.info("Unzipping dataset.")

            # Unzip the file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(directory_path)

            logging.info("Unzip complete.")

            # Remove the temporary zip file
            os.remove(zip_file_path)
        else:
            logging.info(f"Failed to download zip from {url}. HTTP Status Code: {response.status_code}")
    else:
        logging.info(f"The directory {directory_path} already exists.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and combine mask images.')
    parser.add_argument('-i', '--input_path', type=str, help='Path to the input directory containing mask images.')
    parser.add_argument('-d', '--image_dirname', default="", type=str, help='Name of image directory.')
    parser.add_argument('-m', '--mask_dirname', default="masks", type=str, help='Name of image directory.')
    parser.add_argument('-o', '--output_dir', type=str, help='Path to the output directory to save combined masks.')
    parser.add_argument('-v', '--val_pattern', type=str, default='ST_I06', help='String pattern to identify validation files.')
    parser.add_argument('-t', '--test_pattern', type=str, default='ST_C03', help='String pattern to identify test files.')
    parser.add_argument('-f', '--format', type=str, default='png', help='File formats to save the masks.')
    
    args = parser.parse_args()

    setup_logging()

    if os.path.exists(args.output_dir):
        logging.info(f"The output path {args.output_dir} already exists.")
        exit()

    download_and_unzip("https://datasets.simula.no/downloads/cellular-experiments.zip", args.input_path)
        
    process_masks(args.input_path, args.image_dirname, args.mask_dirname, args.output_dir, args.val_pattern, args.test_pattern, args.format)
