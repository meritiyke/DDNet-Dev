#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright:    WZP
Filename:     deformationDetection.py
Description:  Dataset script for Deformation Detection Network (DDNet) with full-sized interferograms.
"""

import random
import numpy as np
from torch.utils.data import Dataset
import os
from torch.utils.data import Dataset # New line added
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def normalize(img):
    """
    Normalize the deformation data to the range [0, 1].
    """
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize to [0, 1]
    return img

class DeformationDetectionDataSet(Dataset):
    def __init__(self, root='', list_path='', max_iters=None,
                 crop_size=None, mirror=True):
        super().__init__()
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size  # Optional cropping (None for full size)
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = os.path.join(self.root, 'interf', name)  # Interferogram data
            label_file = os.path.join(self.root, 'deformation', name)  # Normalized deformation data
            self.files.append({
                "img": img_file,
                "deformation": label_file,
                "name": name
            })

        print("Length of dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        img = np.fromfile(datafiles["img"], dtype=np.float32)  # Load interferogram
        deformation = np.fromfile(datafiles["deformation"], dtype=np.float32)  # Load normalized deformation

        # Reshape to 2D (assuming square interferograms)
        size = int(np.sqrt(img.shape[0]))  # Infer size from data
        image = img.reshape(size, size)
        label = deformation.reshape(size, size)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Shape: (1, H, W)
        label = np.expand_dims(label, axis=0)  # Shape: (1, H, W)

        # Optional cropping
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            img_h, img_w = label.shape[1], label.shape[2]
            assert crop_h <= img_h and crop_w <= img_w, "crop_size is too large"

            h_off = random.randint(0, img_h - crop_h)
            w_off = random.randint(0, img_w - crop_w)
            image = image[:, h_off: h_off + crop_h, w_off: w_off + crop_w]
            label = label[:, h_off: h_off + crop_h, w_off: w_off + crop_w]

        # Random mirroring
        if self.is_mirror:
            fliplr = np.random.choice(2) * 2 - 1
            flipud = np.random.choice(2) * 2 - 1

            image = image[:, ::flipud, ::fliplr]
            label = label[:, ::flipud, ::fliplr]

        # Normalize the interferogram data
        image = normalize(image)
        return image.copy(), label.copy(), np.array(image.shape), datafiles["name"]


class DeformationDetectionValDataSet(Dataset):
    def __init__(self, root='', list_path=''):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, 'interf', name)  # Interferogram data
            label_file = os.path.join(self.root, 'deformation', name)  # Normalized deformation data
            self.files.append({
                "img": img_file,
                "deformation": label_file,
                "name": name
            })

        print("Length of validation dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        img = np.fromfile(datafiles["img"], dtype=np.float32)  # Load interferogram
        deformation = np.fromfile(datafiles["deformation"], dtype=np.float32)  # Load normalized deformation

        # Reshape to 2D (assuming square interferograms)
        size = int(np.sqrt(img.shape[0]))  # Infer size from data
        image = img.reshape(size, size)
        label = deformation.reshape(size, size)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Shape: (1, H, W)
        label = np.expand_dims(label, axis=0)  # Shape: (1, H, W)

        # Normalize the interferogram data
        image = normalize(image)
        return image.copy(), label.copy(), np.array(image.shape), datafiles["name"]


class DeformationDetectionTestDataSet(Dataset):
    def __init__(self, root='', list_path=''):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, 'interf', name)  # Interferogram data
            self.files.append({
                "img": img_file,
                "name": name
            })
        print("Length of test dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        img = np.fromfile(datafiles["img"], dtype=np.float32)  # Load interferogram

        # Reshape to 2D (assuming square interferograms)
        size = int(np.sqrt(img.shape[0]))  # Infer size from data
        image = img.reshape(size, size)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Shape: (1, H, W)

        # Normalize the interferogram data
        image = normalize(image)
        return image.copy(), np.array(image.shape), datafiles["name"]


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Test the dataset
    dataset = DeformationDetectionDataSet(root='./', list_path='./deformationDetection/train.txt', max_iters=None,
                                         crop_size=None, mirror=True)  # Set crop_size=None for full size
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print("Number of batches in dataloader: ", len(dataloader))