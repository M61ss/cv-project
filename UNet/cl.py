import os

import torch

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