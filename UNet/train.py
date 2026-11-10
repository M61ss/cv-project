from math import ceil
import logging
import sys

import torch
from torch.optim import Adam
import monai
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import wandb

from openeds import train_dl, val_dl, test_dl
from earlystopper import EarlyStopping


wandb.login()
wandb_project_name = "UNet"
config = {
    "dataset": "OpenEDS-blur-augmented",
    'architecture': 'UNet',
    'input_channels': 1,
    'output_channels': 4,
    'channels': (8, 16, 32, 64, 128),
    'strides': (2, 2, 2, 2),
    'num_res_units': 3,
    'dropout': 0.4,
    'learning_rate': 1e-4,
    "epoches": 100
}
run = wandb.init(project=wandb_project_name, config=config)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
monai.config.print_config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(
    spatial_dims=2,
    in_channels=config['input_channels'],
    out_channels=config['output_channels'],
    channels=config['channels'],
    strides=config['strides'],
    num_res_units=config['num_res_units'],
    dropout=config['dropout']
).to(device)

loss_fun = DiceLoss(
    softmax=True
)

optimizer = Adam(model.parameters(), lr=config['learning_rate'])

best_val_loss = float('inf')
best_val_loss_epoch = -1

early_stop = EarlyStopping(delta=0.001, patience=10, verbose=True)

for epoch in range(config['epoches']):
    print("=" * 20)
    print(f"EPOCH {epoch + 1}/{config['epoches']}")
            
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

        if early_stop.stop_training:
            print('#' * 20)
            print(f"EARLY STOPPING REACHED at epoch {epoch}")
            print('#' * 20)
            break

print(f"TRAIN COMPLETED!")
print(f"best_metric: {best_val_loss:.4f} at epoch: {best_val_loss_epoch}")
print('\n')
print('@' * 20)
print('Proceed with test...')
print('\n\n')


dice_metric = DiceMetric(
    num_classes=model.out_channels
)

with torch.no_grad():
    model.eval()    
    
    test_num_batches = ceil(len(test_dl.dataset) / float(test_dl.batch_size))
    test_loss = 0

    for test_batch in test_dl:
        test_imgs = test_batch['img'].to(device)
        test_masks = test_batch['mask'].to(device)
        test_labels = test_batch['label'].permute(0, 1, 3, 2).to(device)
        pred_labels = model(test_imgs)
        loss = loss_fun(pred_labels, test_labels)
        dice_metric(pred_labels, test_labels)
        test_loss += loss.item()

    test_loss /= test_num_batches
    print('-' * 20)
    print(f"Average test loss: {test_loss:.4f}")
    print('-' * 20)
    print(f'Dice metric: {dice_metric.aggregate(reduction="mean").item():.4f}')
    print('-' * 20)

wandb.finish()