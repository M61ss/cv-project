import logging
import sys
import os
from glob import glob

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import monai
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data import Dataset, DataLoader, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceLoss

monai.config.print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

checkpoint_dir = os.path.join(os.environ['HOME'], 'cv-project/' + os.path.basename(__file__) + '_checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

train_dir = '/work/cvcs2026/LZMM/OpenEDS/openEDS/openEDS/train'

images = sorted(glob(os.path.join(train_dir, 'images/*.png')))
masks = sorted(glob(os.path.join(train_dir, 'masks/*.png')))

train_files = [ {'img': img, 'mask': mask} for img, mask in zip(images, masks) ]

train_transforms = Compose(
    [
        LoadImaged(keys=['img', 'mask']),
        EnsureChannelFirstd(keys=['img', 'mask']),
    ]
)

train_dataset = Dataset(
    data=train_files,
    transform=train_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
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

optimizer = Adam(model.parameters(), lr=1e-3)

writer = SummaryWriter()

N_EPOCHES = 200

best_loss = float('inf')
for epoch in range(N_EPOCHES):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{N_EPOCHES}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch in train_loader:
        step += 1
        imgs, masks = batch['img'].to(device), batch['mask'].to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = loss_fun(pred, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_dataset) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, os.path.join(checkpoint_dir, 'last.pt'))

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pt'))

writer.close()