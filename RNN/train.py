import torch

from .lstm import LightLSTM


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

model = LightLSTM(
    input_size=1, 
    hidden_size=32, 
    num_layers=1
)