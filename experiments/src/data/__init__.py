import os
import cv2
import glob 

import numpy as np

from PIL import Image
from torch.utils.data import Dataset as BaseDataset


class Cellular(BaseDataset):

    def __init__(self, root_dir, classes=None, augmentation=None, preprocessing=None, binary=False, *args, **argv):

        self.image_paths = sorted(list(glob.glob(os.path.join(root_dir, "images", "*_w1.TIF"))))
        self.mask_paths = sorted(list(glob.glob(os.path.join(root_dir, "masks", "*"))))

        print(len(self.image_paths), len(self.mask_paths))

        assert len(self.mask_paths) == len(self.image_paths)

        self.is_binary = binary

        self.class_values = [ class_id for _, class_id in classes.items() ]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        
        image = self.read_image(self.image_paths[ i ])
        mask = cv2.imread(self.mask_paths[ i ], 0)
        
        if self.is_binary:
            mask[mask > 0] = 1

        masks = [ (mask == v) for v in self.class_values ]
        mask = np.stack(masks, axis=-1).astype("float")

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample[ "image" ], sample[ "mask" ]
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample[ "image" ], sample[ "mask" ]

        return image, mask

    def __len__(self):
        return len(self.mask_paths)

    def read_image(self, path):
        image_1 = np.array(Image.open(path))
        image_2 = np.array(Image.open(path.replace("_w1", "_w2")))
        image_3 = np.array(Image.open(path.replace("_w1", "_w3")))

        image = np.zeros((image_1.shape[0], image_1.shape[1], 3))

        image[:,:,0] = image_1
        image[:,:,1] = image_2
        image[:,:,2] = image_3

        image /= 65534

        return image