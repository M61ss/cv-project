import logging
import sys
import os

import torch
import torch.nn as nn
from torch.optim import Adam
import wandb

from lstm import LightLSTM
from dataset import train_dataset


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

for epoch in range(num_epochs):
    print("=" * 20)
    print(f"EPOCH {epoch + 1}/{num_epochs}")

    model.train()

    train_loss = 0
    for _, x, y in train_dataset:
        x = x.unsqueeze(-1).unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        opt.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        opt.step()

        train_loss += loss.item()