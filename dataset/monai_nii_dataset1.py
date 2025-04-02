# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from .baseDataSet import Base_DataSet
from .utils import *
from .transform_3d import *
import h5py
import numpy as np
import torch
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

from torch.utils.data.sampler import BatchSampler
class _RepeatSampler(BatchSampler):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        # super().__init__(sampler)
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class AlignDataSet(Base_DataSet):
    '''
    DataSet For unaligned data
    '''
    def __init__(self, opt, split):
        super(AlignDataSet, self).__init__()
        self.opt = opt
        self.ext = '.h5'
        self.datasetfile = split == 'train' and opt.train_datasetfile or split == 'dis' and opt.dis_datasetfile or opt.val_datasetfile
        self.dataset_paths = self.get_dataset_from_txt_file(self.datasetfile)
        self.dataset_paths = sorted(self.dataset_paths)
        self.dataset_size = len(self.dataset_paths)
        self.dir_root = self.get_data_path
        self.data_augmentation = List_Compose([
                    (None, None, None),

                    (Resize_image(size=(opt.ct_channel, opt.fine_size, opt.fine_size)),
                    Resize_image(size=(opt.xray_channel, opt.fine_size_cond, opt.fine_size_cond)),
                    Resize_image(size=(opt.xray_channel, opt.fine_size_cond, opt.fine_size_cond)),),

                    (Limit_Min_Max_Threshold(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]), None, None),

                    (Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
                    Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1]),
                    Normalization(opt.XRAY2_MIN_MAX[0], opt.XRAY2_MIN_MAX[1])),

                    (Normalization_to_range(-1, 1),
                    Normalization_to_range(-1, 1),
                    Normalization_to_range(-1, 1)),

                    (Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
                    Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1]),
                    Normalization_gaussian(opt.XRAY2_MEAN_STD[0], opt.XRAY2_MEAN_STD[1])),

                    # (Get_Key_slice(opt.select_slice_num), None),

                    (ToTensor(), ToTensor(), ToTensor()),
                    ])
        self.data_augmentation_1 = List_Compose([(None,None,None)])

    @property
    def name(self):
        return 'AlignDataSet'

    @property
    def get_data_path(self):
        path = os.path.join(self.opt.dataroot)
        return path

    @property
    def num_samples(self):
        return self.dataset_size
    
    def get_dataset_from_txt_file(self, file_path):
        with open(file_path, 'r') as f:
            content = f.readlines()
            return [i.strip() for i in content]

    def get_image_path(self, root, index_name):
        img_path = os.path.join(root, index_name, 'ct_xray_data'+self.ext)
        assert os.path.exists(img_path), 'Path do not exist: {}'.format(img_path)
        return img_path

    def load_file(self, file_path):
        hdf5 = h5py.File(file_path, 'r')
        ct_data = np.asarray(hdf5['ct'])
        x_ray1 = np.asarray(hdf5['xray1'])
        x_ray2 = np.asarray(hdf5['xray2'])
        x_ray1 = np.expand_dims(x_ray1, 0)
        x_ray2 = np.expand_dims(x_ray2, 0)
        hdf5.close()
        return ct_data, x_ray1, x_ray2

    '''
    generate batch
    '''
    def pull_item(self, item):
        file_path = self.get_image_path(self.dir_root, self.dataset_paths[item])
        ct_data, x_ray1, x_ray2 = self.load_file(file_path)

        # print(self.dataset_paths[item])

        # Data Augmentation
        ct, xray1, xray2 = self.data_augmentation([ct_data, x_ray1, x_ray2])
        ct = ct.unsqueeze(0)

        tmp = {'image': ct, 'cond1': xray1, 'cond2': xray2, 'filename': self.dataset_paths[item]}

        return tmp






