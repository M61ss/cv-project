import os
from glob import glob
from random import shuffle

import torch

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd
from monai.data import Dataset, DataLoader, list_data_collate


train_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/train'
validation_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/validation'

train_images = sorted(glob(os.path.join(train_dir, 'images/*.png')))
train_masks = sorted(glob(os.path.join(train_dir, 'masks/*.png')))
validation_images = sorted(glob(os.path.join(validation_dir, 'images/*.png')))
validation_masks = sorted(glob(os.path.join(validation_dir, 'masks/*.png')))

train_files = [ {'img': img, 'mask': mask} for img, mask in zip(train_images, train_masks) ]
validation_files = [ {'img': img, 'mask': mask} for img, mask in zip(validation_images, validation_masks) ]

train_transforms = Compose(
    [
        LoadImaged(keys=['img', 'mask']),
        EnsureChannelFirstd(keys=['img', 'mask']),
        ScaleIntensityd(keys=['img']),
    ]
)
validation_transforms = Compose(
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
validation_dataset = Dataset(
    data=validation_files,
    transform=validation_transforms,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=4,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)