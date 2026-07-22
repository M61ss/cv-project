import logging
import sys
import os
from math import ceil

import torch
from torch.optim import Adam
from monai.networks.nets import UNet
from monai.losses import DiceLoss

import monai

import wandb

from dataset import train_loader, validation_loader


monai.config.print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

LR = 1e-4
N_EPOCHES = 30

loss_fun = DiceLoss(
    sigmoid=True
)

optimizer = Adam(model.parameters(), lr=LR)

wandb.login()
wandb_project_name = "UNet"
wandb_config = {
    "learning_rate": LR,
    "architecture": "UNet",
    "dataset": "OpenEDS",
    "epochs": N_EPOCHES,
}

best_validation_loss = float('inf')
best_validation_loss_epoch = -1

with wandb.init(project=wandb_project_name, config=wandb_config) as run:
    for epoch in range(N_EPOCHES):
        print("=" * 20)
        print(f"EPOCH {epoch + 1}/{N_EPOCHES}")
            
        train_num_batches = ceil(len(train_loader.dataset) / float(train_loader.batch_size))
        validation_num_batches = ceil(len(validation_loader.dataset) / float(validation_loader.batch_size))

        train_loss = 0
        validation_loss = 0

        for i, train_batch in enumerate(train_loader):
            train_imgs, train_masks = train_batch['img'].to(device), train_batch['mask'].to(device)
            optimizer.zero_grad()
            pred = model(train_imgs)
            loss = loss_fun(pred, train_masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"\t{i}/{train_num_batches} train_loss: {loss.item():.4f}")

        train_loss /= train_num_batches
        print(f"\tAverage train loss: {train_loss:.4f}")
        run.log({'train_loss': train_loss})

        with torch.no_grad():
            for validation_batch in validation_loader:
                validation_imgs, validation_masks = validation_batch['img'].to(device), validation_batch['mask'].to(device)
                pred = model(validation_imgs)
                loss = loss_fun(pred, validation_masks)
                validation_loss += loss.item()

            validation_loss /= validation_num_batches
            print(f"\tAverage validation loss: {validation_loss:.4f}")
            run.log({'validation_loss': validation_loss})

            # ADD HERE EARLY STOPPING

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_loss_epoch = epoch + 1

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'validation_loss': validation_loss,
                }, os.path.join(checkpoint_dir, 'best.pth'))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'validation_loss': validation_loss,
            }, os.path.join(checkpoint_dir, 'last.pth'))

    print(f"TRAIN COMPLETED!")
    print(f"best_metric: {best_validation_loss:.4f} at epoch: {best_validation_loss_epoch}")
