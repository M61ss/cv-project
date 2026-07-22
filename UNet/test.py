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

print(torch.unique(test_dataset[sample_idx]['mask']))

with torch.no_grad():
    pred_mask = model(test_img.unsqueeze(0))

pred_mask = pred_mask.squeeze(0).cpu()
test_img = test_img.permute(1, 2, 0).cpu()
test_mask = test_mask.permute(1, 2, 0).cpu()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes[0, 0].imshow(test_img, cmap='gray')
axes[0, 0].set_title('Original Eye Image', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(pred_mask.squeeze(0))
axes[0, 1].set_title('Predicted Mask', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(test_mask.squeeze(0))
axes[0, 2].set_title('True Mask', fontsize=14, fontweight='bold')
axes[0, 2].axis('off')

axes[1, 0].axis('off')

axes[1, 1].imshow(test_img, cmap='gray')
axes[1, 1].imshow(pred_mask.squeeze(0), alpha=0.5)
axes[1, 1].set_title('Image + Predicted Mask', fontsize=14, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(test_img, cmap='gray')
axes[1, 2].imshow(test_mask.squeeze(0), alpha=0.5)
axes[1, 2].set_title('Image + True Mask', fontsize=14, fontweight='bold')
axes[1, 2].axis('off')

plt.savefig(os.path.join(os.path.dirname(__file__), 'plot.png'), dpi=150, bbox_inches='tight')
plt.close(fig)