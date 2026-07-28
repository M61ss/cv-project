import os
from math import ceil

import torch

from monai.networks.nets import UNet
from monai.losses import DiceLoss

from dynunet.openeds import test_dl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(os.path.join(os.path.dirname(__file__), f'checkpoints/best-.pth'))

model = UNet(
    sspatial_dims=2,
    in_channels=1,
    out_channels=4,
    strides=(1, 2, 2, 2),
    kernel_size=(3, 3, 3),
    upsample_kernel_size=(2, 2),
    res_block=True,
    dropout=0.4
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
    print('@' * 20)
    print(f"Average test loss: {test_loss:.4f}")
    print('@' * 20)