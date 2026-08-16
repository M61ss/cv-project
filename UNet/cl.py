import os

import torch

import numpy as np

import h5py

from monai.networks.nets import UNet

from openeds import train_dl


device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(os.path.join(os.path.dirname(__file__), f'checkpoints/blur-mask-aug.pth'))

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,
    channels=(8, 16, 32, 64, 128),
    strides=(2, 2, 2, 2),
    num_res_units=3,
    dropout=0.0
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

with h5py.File('/work/cvcs2026/LZMM/COLET/data_v3.mat', 'r') as f:
    ref_subject = f['Data']['subject_info'][0, 0]
    ref_task = f['Data']['task'][0, 0]
    
    obj_subject = f[ref_subject]
    obj_task = f[ref_task]
    
    print("subject_info[0]:", obj_subject)
    print("task[0]:", obj_task)
    