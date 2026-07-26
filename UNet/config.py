import os
import logging
import sys

import torch

import monai

import wandb


wandb.login()
wandb_project_name = "UNet"
wandb_config = {
    "dataset": "OpenEDS-blur-augmented",
    'architecture': 'UNet',
    'input_channels': 1,
    'output_channels': 4,
    'channels': (8, 16, 32, 64, 128),
    'strides': (2, 2, 2, 2),
    'num_res_units': 3,
    'dropout': 0.4,
    'learning_rate': 1e-4,
    "epoches": 100
}

run = wandb.init(project=wandb_project_name, config=wandb_config)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

monai.config.print_config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
