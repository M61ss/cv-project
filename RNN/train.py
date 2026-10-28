import torch
import torch.nn as nn
from torch.optim import Adam

from lstm import LightLSTM
from dataset import train_dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

model = LightLSTM(
    input_size=1, 
    hidden_size=32, 
    num_layers=1
)

num_epochs = 20
lr = 0.001

loss_fn = nn.CrossEntropyLoss()
opt = Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()

    for _, x, y in train_dataset:
        opt.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        opt.step()