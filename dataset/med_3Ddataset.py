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
import SimpleITK as sitk
import torch.fft as fft
import torch.functional as F


def load_data(config):
    # if not config.data_path:
    #     raise ValueError("unspecified data directory")
    if config.name == "3d-dsa":
        dataset = DSADataset(config)
    else:
        dataset = ImageDataset(config)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    while True:
        yield from loader


def get_number_from_filename(file_path):
    file_name = file_path
    return int(re.search(r"\d+$", file_name).group())


def lowpass_torch(input, limit):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < limit
    pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < limit
    kernel = torch.outer(pass2, pass1)
    fft_input = fft.rfftn(input)
    return fft.irfftn(fft_input * kernel, s=input.shape[-3:])


def highpass_torch(input, limit):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) > limit
    pass2 = torch.abs(fft.fftfreq(input.shape[-2])) > limit
    kernel = torch.outer(pass2, pass1)
    fft_input = fft.rfftn(input)
    return fft.irfftn(fft_input * kernel, s=input.shape[-3:])


def _high_low_loss(rec, target):
    rec_low = lowpass_torch(rec, 0.1)
    target_low = lowpass_torch(target, 0.1)
    rec_high = highpass_torch(rec, 0.05)
    target_high = highpass_torch(target, 0.05)
    return F.mse_loss(rec_low, target_low) + F.mse_loss(rec_high, target_high)


class ImageDataset(Dataset):
    def __init__(self, config, split="train",suffix="DCM") -> None:
        super().__init__()
        # self.image_size = config.image_size
        self.config = config
        self.split = split
        self.suffix = suffix

        # assert "data_path" in config.keys(), "data_path is not in config"
        if "data_path" in self.config.keys():
            self.data_path = config.data_path
            self.data_list = list(sorted(Path(self.data_path).glob("*"), key=lambda x: get_number_from_filename(x.stem)))

        if "cond_path" in self.config.keys():
            self.cond_path = config.cond_path
            self.cond_list = list(sorted(Path(self.cond_path).glob("*"), key=lambda x: get_number_from_filename(x.stem)))

        if split == "train":
            start = 0
            end = int(len(self.data_list) * 0.7)
        elif split == "val":
            start = int(len(self.data_list) * 0.7)
            end = int(len(self.data_list) * 0.8)
        elif split == "test":
            start = int(len(self.data_list) * 0.8)
            end = len(self.data_list)

        self.data_list = self.data_list[start:end]
        if "cond_path" in self.config.keys():
            self.cond_list = self.cond_list[start:end]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index) -> Any:
        path = self.data_list[index]
        source_data = self.preprocess(path, resize_size=self.config.image_size, pad_channel=self.config.pad_channel,suffix=self.suffix)

        assert source_data.shape == (
            1,
            self.config.pad_channel,
            self.config.image_size,
            self.config.image_size,
        ), f"source_data.shape is not correct. {source_data.shape}"

        if "cond_path" in self.config.keys():
            cond_path = self.cond_list[index]
            cond_data = self.preprocess(cond_path, resize_size=self.config.image_size, pad_channel=self.config.pad_channel)
            cond_dict = {"cond": cond_data}
            return source_data, cond_dict
        else:
            return source_data

    def preprocess(self, path, resize_size, pad_channel,suffix):
        dcm_list = list(sorted(Path(path).glob(f"*.{suffix}"), key=lambda x: get_number_from_filename(x.stem)))
        stack_arr = []
        for index, i in enumerate(dcm_list):
            dcm_data = pydicom.dcmread(i)
            if index == 0:
                tags = self.read_tags(dcm_data)

            array = dcm_data.pixel_array
            array = self.normalization(tags, array)

            array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)  # 先变换到0-255的领域里

            if array.shape != (resize_size, resize_size):
                image = Image.fromarray(array, mode="L")
                image = image.resize((self.config.image_size, self.config.image_size), resample=Image.Resampling.BICUBIC)
                array = np.array(image)

                # print(f"image resize to {array.shape}")

            array = array / 127.5 - 1  # 变换到[-1,1]
            array = array.astype(np.float32)

            stack_arr = array if index == 0 else np.dstack((stack_arr, array))
            if index == (pad_channel - 1):
                break
        stack_arr = stack_arr.transpose(2, 0, 1)  # * (512,512,25)->(25,512,512)
        if stack_arr.shape[0] < pad_channel:
            stack_arr = np.pad(
                stack_arr, ((0, pad_channel - stack_arr.shape[0]), (0, 0), (0, 0)), mode="constant", constant_values=0
            )

        assert (
            stack_arr.shape[0] == pad_channel
        ), f"stack_arr.shape[0] is not equal to pad_channel. {stack_arr.shape[0]} != {pad_channel}"

        return stack_arr[None, ::]

    def read_tags(self, file):
        tags = {}
        tags["window_center"] = file.WindowCenter if isinstance(file.WindowCenter, float) else file.WindowCenter[0]
        tags["window_width"] = file.WindowWidth if isinstance(file.WindowWidth, float) else file.WindowWidth[0]

        if hasattr(file, "RescaleSlope"):
            tags["rescale_slope"] = file.RescaleSlope
        if hasattr(file, "RescaleIntercept"):
            tags["rescale_intercept"] = file.RescaleIntercept

        return tags

    def normalization(self, tags, arr):
        """
        :param tags: dicom tags
        :param arr: dicom pixel array
        :return: normalized dicom pixel array (range [0,1])

        """
        center = tags["window_center"]
        width = tags["window_width"]
        dcm_arr = arr
        if "rescale_slope" in tags.keys() and "rescale_intercept" in tags.keys():
            slope = tags["rescale_slope"]
            intercept = tags["rescale_intercept"]
            dcm_arr = arr * slope + intercept

        # 新的方法
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5

        dcm_norm = (dcm_arr - win_min) / (win_max - win_min)

        return dcm_norm

    def reverse_normalization(tags, dcm_arr):
        center = tags["window_center"]
        width = tags["window_width"]
        if "rescale_slope" in tags.keys() and "rescale_intercept" in tags.keys():
            slope = tags["rescale_slope"]
            intercept = tags["rescale_intercept"]

        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5

        dcm_arr = np.clip(dcm_arr, -1, 1)
        dcm_arr = (dcm_arr + 1) * 127.5  # [0-255]
        dcm_arr = np.clip(dcm_arr, 0, 255).astype(np.uint8)

        return dcm_arr

    def draw(self, array):
        pass


class DSADataset(ImageDataset):
    def __init__(self, config) -> None:
        super().__init__(config)
        if config.load_from_csv:
            self.csv_path = config.csv_path
            self.data_list = pd.read_csv(config.csv_path)["location"].tolist()
        else:
            raise NotImplementedError(
                "Only support load dcm from a cvs file recorded every dcm file location, please change LOAD_FROM_CSV true in config \n只支持从csv文件中加载dcm文件，csv文件中记录了每个dcm文件的位置，请在config中将LOAD_FROM_CSV设置为true"
            )

    def __getitem__(self, index) -> Any:
        path = self.data_list[index]
        # * 归一化+Resize Normalization+Resize
        array = self.preprocess(path, self.config.image_size, self.config.clip_channel)
        cond = {}
        return array, cond

    def preprocess(self, path, resize_size, clip_channel):
        dcm = pydicom.dcmread(path)
        tags = self.read_tags(dcm)
        dcm = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(dcm)

        # array = dcm.pixel_array

        # * 归一化至[0,1] normalization to [0,1]
        array = self.normalization(arr=array, tags=tags)

        # # * resize为[clip_channel,image_size,image_size] resize to [clip_channel,image_size,image_size]
        array = array * 255
        array = array.astype(np.uint8)  # * [0,255]

        slice_len = array.shape[0]
        iter_slice = int(0.2 * slice_len)  # * start at 20% 从20%开始切片
        index = 0

        while index < clip_channel:
            image = Image.fromarray(array[iter_slice], mode="L")
            image = image.resize((resize_size, resize_size), resample=3)

            resize_slice = np.array(image)
            if index == 0:
                stack_slice = resize_slice
            else:
                stack_slice = np.dstack((stack_slice, resize_slice))

            index += 1
            iter_slice += 1

        stack_slice = stack_slice.transpose(2, 0, 1)  # * [clip_channel,resize_size,resize_size]
        image = sitk.GetImageFromArray(stack_slice)
        sitk.WriteImage(image, "./1.mhd")
        stack_slice = (stack_slice / 127.5) - 1
        stack_slice = stack_slice.astype(np.float32)

        return stack_slice[None, ::]  # *  [1,clip_channel,resize_size,resize_size]


@hydra.main(version_base=None, config_path="../conf", config_name="/config/autoencoder.yaml")
def main(config: DictConfig):
    config = config["config"]
    dataset = ImageDataset(config)
    for i in dataset:
        # print(i.shape)
        i = (i + 1) * 127.5
        # print(i.max())
        i = i.astype(np.uint8).squeeze()

        # i = i.transpose(0, 2, 3, 1)
        print(i.shape)
        image_ori = sitk.GetImageFromArray(i)
        sitk.WriteImage(image_ori, "./ori.mhd")

        i = torch.tensor(i)
        high = highpass_torch(i, 0.04)
        low = lowpass_torch(i, 0.3)
        # print(high.shape)
        # print(low.shape)
        high = high.numpy()
        high_image = sitk.GetImageFromArray(high)
        sitk.WriteImage(high_image, "./high.mhd")
        low = low.numpy()
        low_image = sitk.GetImageFromArray(low)
        sitk.WriteImage(low_image, "./low.mhd")
        break


if __name__ == "__main__":
    main()
