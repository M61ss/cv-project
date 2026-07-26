import os
from math import ceil

import torch
from torch.optim import Adam
from monai.networks.nets import UNet
from monai.losses import DiceLoss

from config import run, device, checkpoint_dir
from dataset import train_dl, val_dl
from earlystopper import EarlyStopping


model = UNet(
    spatial_dims=2,
    in_channels=run.config['input_channels'],
    out_channels=run.config['output_channels'],
    channels=run.config['channels'],
    strides=run.config['strides'],
    num_res_units=run.config['num_res_units'],
    dropout=run.config['dropout']
).to(device)

loss_fun = DiceLoss(
    softmax=True
)

optimizer = Adam(model.parameters(), lr=run.config['learning_rate'])

best_val_loss = float('inf')
best_val_loss_epoch = -1

early_stop = EarlyStopping(delta=0.001, patience=10, verbose=True)

with run:
    for epoch in range(run.config['epoches']):
        print("=" * 20)
        print(f"EPOCH {epoch + 1}/{run.config['epoches']}")
            
        train_num_batches = ceil(len(train_dl.dataset) / float(train_dl.batch_size))
        val_num_batches = ceil(len(val_dl.dataset) / float(val_dl.batch_size))

        train_loss = 0
        val_loss = 0

        model.train()
        for i, train_batch in enumerate(train_dl):
            train_imgs = train_batch['img'].to(device)
            train_masks = train_batch['mask'].to(device)
            train_labels = train_batch['label'].permute(0, 1, 3, 2).to(device)

            optimizer.zero_grad()
            pred = model(train_imgs)
            loss = loss_fun(pred, train_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"\t{i}/{train_num_batches} train_loss: {loss.item():.4f}")

        train_loss /= train_num_batches
        print(f"\tAverage train loss: {train_loss:.4f}")
        run.log({'train_loss': train_loss})

        with torch.no_grad():
            model.eval()
            for val_batch in val_dl:
                val_imgs = val_batch['img'].to(device)
                val_masks = val_batch['mask'].to(device)
                val_labels = val_batch['label'].permute(0, 1, 3, 2).to(device)

                pred = model(val_imgs)
                loss = loss_fun(pred, val_labels)
                val_loss += loss.item()

            val_loss /= val_num_batches
            print(f"\tAverage validation loss: {val_loss:.4f}")
            run.log({'validation_loss': val_loss})

            early_stop.check_early_stop(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch + 1

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'validation_loss': val_loss,
                }, os.path.join(checkpoint_dir, f'best-{run.name}.pth'))

            if early_stop.stop_training:
                print('#' * 20)
                print(f"EARLY STOPPING REACHED at epoch {epoch}")
                print('#' * 20)
                break

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'validation_loss': val_loss,
            }, os.path.join(checkpoint_dir, f'last-{run.name}.pth'))

    print(f"TRAIN COMPLETED!")
    print(f"best_metric: {best_val_loss:.4f} at epoch: {best_val_loss_epoch}")
