import logging
import sys
import os
from glob import glob

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import monai
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd
from monai.data import Dataset, DataLoader, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceLoss


def run(rank, size):
    pass


def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    world_size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


checkpoint_dir = os.path.join(os.environ['HOME'], 'cv-project/UNet-OpenEDS-supervised_checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

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
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=torch.cuda.is_available(),
    collate_fn=list_data_collate,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
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

writer = SummaryWriter()

N_EPOCHES = 100

best_validation_loss = float('-inf')
best_validation_loss_epoch = -1
for epoch in range(N_EPOCHES):
    print("=" * 20)
    print(f"EPOCH {epoch + 1}/{N_EPOCHES}")

    model.train()
    epoch_train_loss = 0
    train_steps = 0
    for train_batch in train_loader:
        train_steps += 1
        train_imgs, train_masks = train_batch['img'].to(device), train_batch['mask'].to(device)
        optimizer.zero_grad()
        pred = model(train_imgs)
        loss = loss_fun(pred, train_masks)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        epoch_train_len = len(train_dataset) // train_loader.batch_size
        print(f"\t{train_steps}/{epoch_train_len} train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_train_len * epoch + train_steps)
    epoch_train_loss /= train_steps
    print(f"\tAverage train loss: {epoch_train_loss:.4f}")

    model.eval()
    with torch.no_grad():
        epoch_validation_loss = 0
        validation_steps = 0
        for validation_batch in validation_loader:
            validation_steps += 1
            validation_imgs, validation_masks = validation_batch['img'].to(device), validation_batch['mask'].to(device)
            pred = model(validation_imgs)
            loss = loss_fun(pred, validation_masks)
            epoch_validation_loss += loss.item()
        epoch_validation_loss /= validation_steps
        print(f"\tAverage validation loss: {epoch_validation_loss:.4f}")

        # ADD EARLY STOPPING

        if epoch_validation_loss > best_validation_loss:
            best_validation_loss = epoch_validation_loss
            best_validation_loss_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'validation_loss': epoch_validation_loss,
            }, os.path.join(checkpoint_dir, 'best.pth'))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_train_loss,
        'validation_loss': epoch_validation_loss,
    }, os.path.join(checkpoint_dir, 'last.pth'))

print(f"TRAIN COMPLETED!")
print(f"best_metric: {best_validation_loss:.4f} at epoch: {best_validation_loss_epoch}")
writer.close()