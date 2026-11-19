import torch
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


class MILLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        BCH, BAG, _ = x.shape
        h_n_tot = torch.tensor(0)
        for i in range(BAG):
            instance = x[:, i]
            out, (h_n, c_n) = self.lstm(instance)
            h_n_tot = torch.cat([h_n_tot, h_n.unsqueeze(1)], dim=1)
        h_n_tot = torch.sum(h_n_tot, dim=1) / BAG
        h_final = self.fc(h_n_tot[-1])
        return h_final