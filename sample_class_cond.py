from tqdm import tqdm
from pathlib import Path
import util.misc as misc
from util.shapenet import ShapeNet, category_ids

import models_ae as models_ae

import mcubes
import trimesh
from scipy.spatial import cKDTree as KDTree
import numpy as np
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch
import yaml
import math

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Diffusion', type=str,
                    metavar='MODEL', help='Name of model to train')
parser.add_argument(
    '--pth', default='output/checkpoint-130.pth', type=str)
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()


# import utils


def main():
    print(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = models_ae.__dict__[args.model]()
    device = torch.device(args.device)

    model.eval()
    model.load_state_dict(torch.load(args.pth, map_location='cpu')[
                          'model'], strict=True)
    model.to(device)
    # print(model)

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].cuda()

    with torch.no_grad():
        for idx in range(16):

            categories = torch.Tensor([0] * 1).int().cuda()
            outputs = model.sample(categories, grid.expand(1, -1, -1), n_steps=64)

            output = outputs[0]
            volume = output.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy() * (-1)

            verts, faces = mcubes.marching_cubes(volume, 0)
            verts *= gap
            verts -= 1.
            m = trimesh.Trimesh(verts, faces)

            m.export('samples/{:03d}.obj'.format(idx))

if __name__ == '__main__':
    main()
