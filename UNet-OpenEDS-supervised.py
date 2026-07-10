import logging
import sys
import os
from glob import glob

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import monai
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd
from monai.data import Dataset, DataLoader, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

monai.config.print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

checkpoint_dir = os.path.join(os.environ['HOME'], 'cv-project/UNet-OpenEDS-supervised_checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

train_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/train'
test_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/test'

train_images = sorted(glob(os.path.join(train_dir, 'images/*.png')))
train_masks = sorted(glob(os.path.join(train_dir, 'masks/*.png')))
test_images = sorted(glob(os.path.join(test_dir, 'images/*.png')))
test_masks = sorted(glob(os.path.join(test_dir, 'masks/*.png')))

train_files = [ {'img': img, 'mask': mask} for img, mask in zip(train_images, train_masks) ]
test_files = [ {'img': img, 'mask': mask} for img, mask in zip(test_images, test_masks) ]

train_transforms = Compose(
    [
        LoadImaged(keys=['img', 'mask']),
        EnsureChannelFirstd(keys=['img', 'mask']),
        ScaleIntensityd(keys=['img']),
    ]
)
test_transforms = Compose(
    [
        LoadImaged(keys=['img', 'mask']),
        EnsureChannelFirstd(keys=['img', 'mask']),
        ScaleIntensityd(keys=['img']),
    ]
)

train_dataset = Dataset(
    data=train_files,
    transform=train_transforms,
)
test_dataset = Dataset(
    data=test_files,
    transform=test_transforms,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=16,
    persistent_workers=True,
    prefetch_factor=8,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

loss_fun = DiceLoss(
    sigmoid=True
)

optimizer = Adam(model.parameters(), lr=1e-4)

dice_metric = DiceMetric(
    include_background=True,
    reduction='mean',
    get_not_nans=False,
)

writer = SummaryWriter()

N_EPOCHES = 100

best_metric = float('-inf')
best_metric_epoch = -1
for epoch in range(N_EPOCHES):
    print("=" * 20)
    print(f"EPOCH {epoch + 1}/{N_EPOCHES}")

    model.train()
    epoch_loss = 0
    train_steps = 0
    for train_batch in train_loader:
        train_steps += 1
        train_imgs, train_masks = train_batch['img'].to(device), train_batch['mask'].to(device)
        optimizer.zero_grad()
        pred = model(train_imgs)
        loss = loss_fun(pred, train_masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_train_len = len(train_dataset) // train_loader.batch_size
        print(f"\t{train_steps}/{epoch_train_len} train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_train_len * epoch + train_steps)
    epoch_loss /= train_steps
    print(f"\tAverage train loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        for test_batch in test_loader:
            test_imgs, test_masks = test_batch['img'].to(device), test_batch['mask'].to(device)
            pred = model(test_imgs)
            dice_metric(pred, test_masks)
        metric = dice_metric.aggregate().item()
        print(f"\tDice metric: {metric:.4f}")
        dice_metric.reset()
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(checkpoint_dir, 'best.pth'))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, os.path.join(checkpoint_dir, 'last.pth'))

print(f"TRAIN COMPLETED!")
print(f"best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()