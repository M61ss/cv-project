import os
from math import ceil

import torch

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

from openeds import test_dl


def test_model(model: UNet, loss_fun):
    model.eval()

    dice_metric = DiceMetric(
        num_classes=model.out_channels
    )
    
    with torch.no_grad():
        test_num_batches = ceil(len(test_dl.dataset) / float(test_dl.batch_size))
        test_loss = 0
        for test_batch in test_dl:
            test_imgs = test_batch['img'].to(device)
            test_masks = test_batch['mask'].to(device)
            test_labels = test_batch['label'].permute(0, 1, 3, 2).to(device)

            pred_labels = model(test_imgs)
            loss = loss_fun(pred_labels, test_labels)
            test_loss += loss.item()

        test_loss /= test_num_batches
        print('-' * 20)
        print(f"Average test loss: {test_loss:.4f}")
        print('-' * 20)
        print('Dice metric:')
        print(f'\t- mean:\t\t{dice_metric.aggregate(reduction="mean")}')
        print(f'\t- mean_batch:\t\t{dice_metric.aggregate(reduction="mean_batch")}')
        print(f'\t- mean_channel:\t{dice_metric.aggregate(reduction="mean_channel")}')
        print('-' * 20)


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

loss_fun = DiceLoss(
    softmax=True
)

test_model(model, loss_fun)