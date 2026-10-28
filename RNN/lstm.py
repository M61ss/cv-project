import torch.nn as nn


class LightLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        h_final = self.fc(h_n[-1])
        return h_final
