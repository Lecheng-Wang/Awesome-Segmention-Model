# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : ${2025/5/15} ${20:17}
# @Function: Compress images and labels to a datasets for model input


import os
import numpy as np
import gdal
import torch
from torch.utils.data.dataset import Dataset
gdal.UseExceptions()

class Labeled_Model_Dataset(Dataset):
    def __init__(self, annotation_lines, dataset_path):
        super(Labeled_Model_Dataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length           = len(annotation_lines)
        self.dataset_path     = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line   = self.annotation_lines[index]
        name              = annotation_line.split()[0]
        image             = gdal.Open(os.path.join(os.path.join(self.dataset_path, "images"), name + ".tif")).ReadAsArray().astype(np.float32)
        image             = np.nan_to_num(image, nan=0.0)
        label             = gdal.Open(os.path.join(os.path.join(self.dataset_path, "labels"), name + ".tif")).ReadAsArray()
        label[label==128] = 1
        label[label==255] = 2
        return image, label

class UnLabeled_Model_Dataset(Dataset):
    def __init__(self, annotation_lines, dataset_path):
        super(UnLabeled_Model_Dataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length           = len(annotation_lines)
        self.dataset_path     = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line   = self.annotation_lines[index]
        name              = annotation_line.split()[0]
        image             = gdal.Open(os.path.join(os.path.join(self.dataset_path, "images"), name + ".tif")).ReadAsArray().astype(np.float32)
        image             = np.nan_to_num(image, nan=0.0)
        return image


