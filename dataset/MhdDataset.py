import math
import random
import hydra
from omegaconf import DictConfig

from PIL import Image
import re
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import sys
from typing import Any, Union, List

import torch
from torch.utils.data import dataloader
import pydicom
import SimpleITK as sitk
import pandas as pd
from matplotlib import pyplot as plt
import time
import torch.fft as fft
import torch.functional as F


class MhdDataset(Dataset):
    def __init__(self, config, split="train", mean=None, std=None):
        self.data_path = sorted(Path(config.data_path).glob("*.mhd"))
        if not mean or not std:
            print("Calculating mean and std")
            self.mean, self.std = self.calculate_mean_std()
        else:
            self.mean = mean
            self.std = std
        print(f"Dataset mean: {self.mean}, std: {self.std}")
        # 80% train 20% test
        if split == "train":
            self.data_path = self.data_path[: int(len(self.data_path) * 0.8)]
        else:
            self.data_path = self.data_path[int(len(self.data_path) * 0.8) :]

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        path = self.data_path[index]
        img = sitk.ReadImage(path)
        img = self.resize_image(img, new_size=(32, 128, 128))
        arr = sitk.GetArrayFromImage(img)
        # arr = arr.transpose(2, 1, 0)
        img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(img, "test.mhd")
        # arr =
        # self.show_mhd(arr)
        arr = (arr - self.mean) / self.std
        arr = arr.astype(np.float32)
        return arr[None, ::]

    def calculate_mean_std(self):
        mean = []
        std = []
        for path in self.data_path:
            img = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(img)
            mean.append(arr.mean())
            std.append(arr.std())
        return np.mean(mean), np.mean(std)

    def show_mhd(self, arr):
        print(f"shape: {arr.shape}")
        print(f"max : {arr.max()}")
        print(f"min : {arr.min()}")
        print(f"mean: {arr.mean()}")

    def resize_image(self, image, new_size):
        original_size = np.array(image.GetSize())
        original_spacing = np.array(image.GetSpacing())

        new_spacing = original_spacing * (original_size / new_size)

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spacing.tolist())
        resample.SetSize(new_size)
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetDefaultPixelValue(image.GetPixelIDValue())

        return resample.Execute(image)


@hydra.main(config_path="../conf", config_name="config/autoencoder.yaml", version_base="1.3")
def main(config):
    config = config["config"]
    ds = MhdDataset(config)
    for i in ds:
        break


if __name__ == "__main__":
    main()
