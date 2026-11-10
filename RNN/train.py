import logging
import sys
import os

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.optim import Adam
import wandb

from lstm import LightLSTM
from dataset import train_dataset, val_dataset, test_dataset, ds_config
from earlystopper import EarlyStopping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.login()
wandb_project_name = "RNN Head"
config = {
    "dataset": "IREye4Task",
    'architecture': 'LSTM',
    'input_dim': 1,
    'output_dim': 1,
    'hidden_dim': 16,
    'num_layers': 4,
    'learning_rate': 1e-4,
    "num_epoches": 50
}
run = wandb.init(project=wandb_project_name, config=(config | ds_config))

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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
early_stop = EarlyStopping(delta=0.001, patience=7, verbose=True)

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

with torch.no_grad():
    model.eval()

    y_pred = torch.empty(0)
    y_true = torch.empty(0)
    test_loss = 0
    for _, x, y in test_dataset:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)
        test_loss += loss.item()
        y_pred = torch.concat([y_pred, torch.round(torch.sigmoid(pred.cpu()))])
        y_true = torch.concat([y_true, torch.round(torch.sigmoid(y.cpu()))])

    print(f'Average test loss:  {test_loss / len(test_dataset)}')
    print(f'Accuracy:           {accuracy_score(y_true, y_pred)}')
    print(f'Precision:          {precision_score(y_true, y_pred)}')
    print(f'Recall:             {recall_score(y_true, y_pred)}')
    print(f'F1:                 {f1_score(y_true, y_pred)}')
    print('Confusion matrix:')
    print(f'{confusion_matrix(y_true, y_pred)}')
