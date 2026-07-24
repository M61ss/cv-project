import os
from glob import glob
import random

import torch
from torch.utils.data import ConcatDataset

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, AsDiscreted, GaussianSmoothd
from monai.data import Dataset, DataLoader, list_data_collate


data_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/train'

images = sorted(glob(os.path.join(data_dir, 'images/*.png')))
masks = sorted(glob(os.path.join(data_dir, 'masks/*.png')))
labels = sorted(glob(os.path.join(data_dir, 'labels/*.npy')))

files = [ {
            'img': img, 
            'mask': mask, 
            'label': label
        } for img, mask, label in zip(images, masks, labels)
]

random.seed(42)
shuffled_files = files.copy()
random.shuffle(shuffled_files)

n = len(shuffled_files)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

train_files = shuffled_files[:n_train]
val_files = shuffled_files[n_train:n_train + n_val]
test_files = shuffled_files[n_train + n_val:]

print('Original train images:\t', len(train_files))
print('Original val images:\t', len(val_files))
print('Original test images:\t', len(test_files))

original_transforms = Compose(
    transforms=[
        LoadImaged(keys=['img', 'mask', 'label']),
        EnsureChannelFirstd(keys=['img', 'mask', 'label']),
        ScaleIntensityd(keys=['img']),
        AsDiscreted(keys=['label'], to_onehot=4)
    ]
)
augmentation_transforms = Compose(
    transforms=[
        LoadImaged(keys=['img', 'mask', 'label']),
        EnsureChannelFirstd(keys=['img', 'mask', 'label']),
        ScaleIntensityd(keys=['img']),
        GaussianSmoothd(keys=['img'], sigma=1),
        AsDiscreted(keys=['label'], to_onehot=4)
    ]
)

train_dataset = ConcatDataset(
    [
        Dataset(
            data=train_files,
            transform=original_transforms
        ),
        Dataset(
            data=train_files,
            transform=augmentation_transforms
        )
    ]
)
val_dataset = ConcatDataset(
    [
        Dataset(
            data=val_files,
            transform=original_transforms
        ),
        Dataset(
            data=val_files,
            transform=augmentation_transforms
        )
    ]    
)
test_dataset = ConcatDataset(
    [
        Dataset(
            data=test_files,
            transform=original_transforms
        ),
        Dataset(
            data=test_files,
            transform=augmentation_transforms
        )
    ]
)

print('Augmented train images:\t', len(train_dataset))
print('Augmented val images:\t', len(val_dataset))
print('Augmented test images:\t', len(test_dataset))

train_dl = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=12,
    persistent_workers=True,
    prefetch_factor=8,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)
val_dl = DataLoader(
    val_dataset,
    batch_size=32,
    num_workers=12,
    persistent_workers=True,
    prefetch_factor=8,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)
# test_dl = DataLoader(
#     test_dataset,
#     batch_size=64,
#     num_workers=12,
#     persistent_workers=False,
#     prefetch_factor=8,
#     pin_memory=torch.cuda.is_available(),
#     collate_fn=list_data_collate,
# )