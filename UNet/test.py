import os
from math import ceil

import torch

from monai.networks.nets import UNet
from monai.losses import DiceLoss

from config import config
from dataset import test_dl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'checkpoints/best.pth'))

model = UNet(
    spatial_dims=2,
    in_channels=config['input_channels'],
    out_channels=config['output_channels'],
    channels=config['channels'],
    strides=config['strides'],
    num_res_units=config['num_res_units'],
    dropout=config['dropout']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

loss_fun = DiceLoss(
    softmax=True
)

with torch.no_grad():
    test_num_batches = ceil(len(test_dl.dataset) / float(test_dl.batch_size))
    test_loss = 0
    pupil_accuracy = 0
    for test_batch in test_dl:
        test_imgs = test_batch['img'].to(device)
        test_masks = test_batch['mask'].to(device)
        test_labels = test_batch['label'].permute(0, 1, 3, 2).to(device)

        pred_labels = model(test_imgs)
        loss = loss_fun(pred_labels, test_labels)
        test_loss += loss.item()

    test_loss /= test_num_batches
    print(f"\tAverage test loss: {test_loss:.4f}")
