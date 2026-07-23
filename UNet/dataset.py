import os
from glob import glob

import torch
from torch.utils.data.dataset import random_split

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd
from monai.data import Dataset, DataLoader, list_data_collate


data_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/train'
# val_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/validation'
# test_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/test'

images = sorted(glob(os.path.join(data_dir, 'images/*.png')))
masks = sorted(glob(os.path.join(data_dir, 'masks/*.png')))
# val_images = sorted(glob(os.path.join(val_dir, 'images/*.png')))
# val_masks = sorted(glob(os.path.join(val_dir, 'masks/*.png')))
# test_images = sorted(glob(os.path.join(test_dir, 'images/*.png')))
# test_masks = sorted(glob(os.path.join(test_dir, 'masks/*.png')))

train_files = [ {'img': img, 'mask': mask} for img, mask in zip(images, masks) ]
# val_files = [ {'img': img, 'mask': mask} for img, mask in zip(val_images, val_masks) ]
# test_files = [ {'img': img, 'mask': mask} for img, mask in zip(test_images, test_masks) ]

print('Total images:\t', len(train_files))
# print('Val files:\t', len(val_files))
# print('Test files:\t', len(test_files))

transforms = Compose(
    [
        LoadImaged(keys=['img', 'mask']),
        EnsureChannelFirstd(keys=['img', 'mask']),
        ScaleIntensityd(keys=['img']),
    ]
)
# val_transforms = Compose(
#     [
#         LoadImaged(keys=['img', 'mask']),
#         EnsureChannelFirstd(keys=['img', 'mask']),
#         ScaleIntensityd(keys=['img']),
#     ]
# )
# test_transforms = Compose(
#     [
#         LoadImaged(keys=['img', 'mask']),
#         EnsureChannelFirstd(keys=['img', 'mask']),
#         ScaleIntensityd(keys=['img']),
#     ]
# )

dataset = Dataset(
    data=train_files,
    transform=transforms,
)
# val_dataset = Dataset(
#     data=val_files,
#     transform=val_transforms,
# )
# test_dataset = Dataset(
#     data=test_files,
#     transform=test_transforms
# )

train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

print('Train images:\t', len(train_dataset))
print('Val images:\t', len(val_dataset))
print('Test images:\t', len(test_dataset))

train_dl = DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)
val_dl = DataLoader(
    val_dataset,
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)
test_dl = DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=8,
    persistent_workers=False,
    prefetch_factor=2,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)