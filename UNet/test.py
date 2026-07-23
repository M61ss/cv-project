import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch

from monai.networks.nets import UNet

from dataset import test_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'checkpoints/best.pth'))

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
test_img = test_dataset[sample_idx]['img'].to(device)
test_mask = test_dataset[sample_idx]['mask'].to(device)
test_label = test_dataset[sample_idx]['label'].to(device)

print('True unique values:\t', torch.unique(test_dataset[sample_idx]['label']))

with torch.no_grad():
    pred_logits = model(test_img.unsqueeze(0))
    pred_prob = torch.sigmoid(pred_logits)
    pred_label = (pred_prob > 0.5).float().squeeze(0)
    print('Pred unique values:\t', torch.unique(pred_label))

print('Image shape:\t\t', test_img.shape)
print('True label shape:\t', test_label.shape)
print('Pred label shape:\t', pred_label.shape)

pred_label = pred_label.argmax(dim=0).permute(1, 0).cpu()
test_img = test_img.permute(2, 1, 0).cpu()
test_label = test_label.argmax(dim=0).permute(0, 1).cpu()

cmap = mcolors.ListedColormap(['black', 'blue', 'green', 'red'])

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes[0, 0].imshow(test_img, cmap='gray')
axes[0, 0].set_title('Original Eye Image', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(pred_label, cmap=cmap, vmin=0, vmax=3)
axes[0, 1].set_title('Predicted Label', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(test_label, cmap=cmap, vmin=0, vmax=3)
axes[0, 2].set_title('True Label', fontsize=14, fontweight='bold')
axes[0, 2].axis('off')

axes[1, 0].axis('off')

axes[1, 1].imshow(test_img, cmap='gray')
axes[1, 1].imshow(pred_label, alpha=0.5)
axes[1, 1].set_title('Image + Predicted Label', fontsize=14, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(test_img, cmap='gray')
axes[1, 2].imshow(test_label, alpha=0.5)
axes[1, 2].set_title('Image + True Label', fontsize=14, fontweight='bold')
axes[1, 2].axis('off')

plt.savefig(os.path.join(os.path.dirname(__file__), 'plot.png'), dpi=150, bbox_inches='tight')
plt.close(fig)