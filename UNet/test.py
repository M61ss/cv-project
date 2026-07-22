import os

import matplotlib.pyplot as plt
import cv2

import torch

from monai.networks.nets import UNet

from dataset import test_dataset
from pltutils import apply_mask

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

with torch.no_grad():
    pred_mask = model(test_img.unsqueeze(0))

pred_mask = pred_mask.squeeze(0).cpu()
test_img = test_img.permute(1, 2, 0).cpu()
test_mask = test_mask.permute(1, 2, 0).cpu()

fig, axes = plt.subplots(1, 3, figsize=(18, 12))
axes[0].imshow(test_img)
axes[0].set_title('Original Eye Image', fontsize=14, fontweight='bold')
axes[0].axis('off')

pred_overlay = apply_mask(test_img, pred_mask)
axes[1].imshow(pred_overlay)
axes[1].set_title('Image + Predicted Mask', fontsize=14, fontweight='bold')
axes[1].axis('off')

true_overlay = apply_mask(test_img, test_mask)
axes[2].imshow(true_overlay)
axes[2].set_title('Image + True Mask', fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.show()