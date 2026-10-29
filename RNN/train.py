import logging
import sys
import os

import torch
import torch.nn as nn
from torch.optim import Adam
import wandb

from lstm import LightLSTM
from dataset import train_dataset, val_dataset
from earlystopper import EarlyStopping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.login()
wandb_project_name = "RNN Head"
config = {
    "dataset": "IREye4Task",
    'architecture': 'LSTM',
    'input_dim': 1,
    'output_dim': 1,
    'hidden_dim': 32,
    'num_layers': 1,
    'learning_rate': 1e-3,
    "num_epoches": 20
}
run = wandb.init(project=wandb_project_name, config=config)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

print('Device:', device)

model = LightLSTM(
    input_size=config['input_dim'],
    out_size=config['output_dim'], 
    hidden_size=config['hidden_dim'], 
    num_layers=config['num_layers']
).to(device)

num_epochs = config['num_epoches']
lr = config['learning_rate']

loss_fn = nn.BCEWithLogitsLoss().to(device)
opt = Adam(model.parameters(), lr=lr)
early_stop = EarlyStopping(delta=0.001, patience=10, verbose=True)

best_val_loss = float('inf')
best_val_loss_epoch = -1

for epoch in range(num_epochs):
    print("=" * 20)
    print(f"EPOCH {epoch + 1}/{num_epochs}")

    model.train()

    train_loss = 0
    for i, train_batch in enumerate(train_dataset):
        x = train_batch[1].to(device)
        y = train_batch[2].to(device)

        opt.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        opt.step()

        train_loss += loss.item()
        print(f"\t{i}/{len(train_dataset)} train_loss: {loss.item():.4f}")

    train_loss /= len(train_dataset)
    print(f"\tAverage train loss: {train_loss:.4f}")
    run.log({'train_loss': train_loss})

    with torch.no_grad():
        model.eval()

        val_loss = 0
        for _, x, y in val_dataset:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

        val_loss /= len(val_dataset)
        print(f"\tAverage validation loss: {val_loss:.4f}")
        run.log({'validation_loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch + 1
        
        early_stop.check_early_stop(val_loss)

        if early_stop.stop_training:
            print('#' * 20)
            print(f"EARLY STOPPING REACHED at epoch {epoch}")
            print('#' * 20)
            break

print(f"TRAINING COMPLETED!")
print(f"best_metric: {best_val_loss:.4f} at epoch: {best_val_loss_epoch}")