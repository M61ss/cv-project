import os

import matplotlib.pyplot as plt

import torch

from monai.networks.nets import UNet

from dataset import test_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'checkpoints/best.pth'))

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
test_img, test_mask = test_dataset[sample_idx]['img'].to(device), test_dataset[sample_idx]['mask'].to(device)

