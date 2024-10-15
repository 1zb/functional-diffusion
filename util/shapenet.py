
import os
import glob
import random

import yaml 

import torch
from torch.utils import data

import numpy as np

from PIL import Image

import h5py

category_ids = {
    '02691156': 0,
    '02747177': 1,
    '02773838': 2,
    '02801938': 3,
    '02808440': 4,
    '02818832': 5,
    '02828884': 6,
    '02843684': 7,
    '02871439': 8,
    '02876657': 9, 
    '02880940': 10,
    '02924116': 11,
    '02933112': 12,
    '02942699': 13,
    '02946921': 14,
    '02954340': 15,
    '02958343': 16,
    '02992529': 17,
    '03001627': 18,
    '03046257': 19,
    '03085013': 20,
    '03207941': 21,
    '03211117': 22,
    '03261776': 23,
    '03325088': 24,
    '03337140': 25,
    '03467517': 26,
    '03513137': 27,
    '03593526': 28,
    '03624134': 29,
    '03636649': 30,
    '03642806': 31,
    '03691459': 32,
    '03710193': 33,
    '03759954': 34,
    '03761084': 35,
    '03790512': 36,
    '03797390': 37,
    '03928116': 38,
    '03938244': 39,
    '03948459': 40,
    '03991062': 41,
    '04004475': 42,
    '04074963': 43,
    '04090263': 44,
    '04099429': 45,
    '04225987': 46,
    '04256520': 47,
    '04330267': 48,
    '04379243': 49,
    '04401088': 50,
    '04460130': 51,
    '04468005': 52,
    '04530566': 53,
    '04554684': 54,
}

class ShapeNet(data.Dataset):
    def __init__(self, dataset_folder, split, categories=None, transform=None, sampling=True, num_samples=4096, return_surface=True, surface_sampling=True, pc_size=2048, replica=16):
        
        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder
        # self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_sdf')
        self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_sdf')
        self.mesh_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_watertight')

        # categories = None
        if categories is None:
            categories = os.listdir(self.point_folder)
            categories = [c for c in categories if os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()
        
        # categories = ['03001627']

        print(categories)

        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.point_folder, c)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath.replace('ShapeNetV2_sdf', 'ShapeNetV2_point'), split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            
            self.models += [
                {'category': c, 'model': m.replace('.npz', '')}
                for m in models_c
            ]

        self.replica = replica

        if self.split == 'train':
            self.hf = []
            # self.accum_sum = [0]
            for i in range(8):
                hf = h5py.File('/ibex/ai/project/c2168/biao/shapenet_sdf_h5/{}-{:03d}.h5'.format(self.split, i), 'r')
                self.hf.append(hf)
                # self.accum_sum.append(self.accum_sum[-1] + len(hf.keys()) // 5)
                # print(self.accum_sum)
        else:
            self.hf = h5py.File('/ibex/ai/project/c2168/biao/shapenet_sdf_h5/{}-000.h5'.format(self.split), 'r')

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        category = self.models[idx]['category']

        model = self.models[idx]['model']

        if isinstance(self.hf, list):
            hf = self.hf[idx % 8]
        else:
            hf = self.hf

        vol_points = hf['{}_{}_{}'.format(category, model, 'vol_points')][:]
        vol_sdf = hf['{}_{}_{}'.format(category, model, 'vol_sdf')][:]
        near_points = hf['{}_{}_{}'.format(category, model, 'near_points')][:]
        near_label = hf['{}_{}_{}'.format(category, model, 'near_sdf')][:]
        surface = hf['{}_{}_{}'.format(category, model, 'surface_points')][:]


        if self.return_surface:
            if self.surface_sampling:
                ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = surface[ind]
            surface = torch.from_numpy(surface)

        ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
        vol_points2 = vol_points[ind]
        vol_sdf2 = vol_sdf[ind]
            
        if self.sampling:
            
            ind = np.random.default_rng().choice(vol_points.shape[0], 1024, replace=False)
            vol_points = vol_points[ind]
            vol_sdf = vol_sdf[ind]

            
            ind = np.random.default_rng().choice(near_points.shape[0], 1024, replace=False)
            near_points = near_points[ind]
            near_label = near_label[ind]

        
        vol_points = torch.from_numpy(vol_points)
        vol_sdf = torch.from_numpy(vol_sdf).float()

        vol_points2 = torch.from_numpy(vol_points2)
        vol_sdf2 = torch.from_numpy(vol_sdf2).float()
        
        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

        
            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_sdf, near_label], dim=0)
        else:

            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

        
            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_sdf, near_label], dim=0)

        if self.transform:
            surface, points = self.transform(surface, points)


        if self.return_surface:
            return points, labels, vol_points2, vol_sdf2, surface, category_ids[category]#, model
        else:
            return points, labels, category_ids[category]

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica
