import torch
from torch.nn import LSTM


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

model = LSTM(1, 16, 2)