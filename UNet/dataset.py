import os
from glob import glob

import torch

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd
from monai.data import Dataset, DataLoader, list_data_collate


train_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/train'
val_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/validation'
test_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/test'

train_images = sorted(glob(os.path.join(train_dir, 'images/*.png')))
train_masks = sorted(glob(os.path.join(train_dir, 'masks/*.png')))
val_images = sorted(glob(os.path.join(val_dir, 'images/*.png')))
val_masks = sorted(glob(os.path.join(val_dir, 'masks/*.png')))
test_images = sorted(glob(os.path.join(test_dir, 'images/*.png')))
test_masks = sorted(glob(os.path.join(test_dir, 'masks/*.png')))

train_files = [ {'img': img, 'mask': mask} for img, mask in zip(train_images, train_masks) ]
val_files = [ {'img': img, 'mask': mask} for img, mask in zip(val_images, val_masks) ]
test_files = [ {'img': img, 'mask': mask} for img, mask in zip(test_images, test_masks) ]

train_transforms = Compose(
    [
        LoadImaged(keys=['img', 'mask']),
        EnsureChannelFirstd(keys=['img', 'mask']),
        ScaleIntensityd(keys=['img']),
    ]
)
val_transforms = Compose(
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
val_dataset = Dataset(
    data=val_files,
    transform=val_transforms,
)
test_dataset = Dataset(
    data=test_files,
    transform=test_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)