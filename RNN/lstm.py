import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LightLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        h_final = self.fc(h_n[-1])
        return h_final


class MILAvgLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        BCH, BAG, *_ = x.shape
        h_n_tot = torch.empty(0, device=device)
        for i in range(BAG):
            instance = x[:, i]
            out, (h_n, c_n) = self.lstm(instance)
            h_n_tot = torch.cat([h_n_tot, h_n.unsqueeze(1)], dim=1)
        h_n_tot = torch.sum(h_n_tot, dim=1) / BAG
        h_final = self.fc(h_n_tot[-1])
        return h_final


class MILAttnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.qW = nn.Linear(hidden_size, hidden_size)
        self.kW = nn.Linear(hidden_size, hidden_size)
        self.vW = nn.Linear(hidden_size, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, 2, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        BCH, BAG, *_ = x.shape
        
        Q_tot = torch.empty(0, device=device)
        K_tot = torch.empty(0, device=device)
        V_tot = torch.empty(0, device=device)
        for i in range(BAG):
            instance = x[:, i]
            out, (h_n, c_n) = self.lstm(instance)
            Q_n = self.qW(h_n)
            K_n = self.kW(h_n)
            V_n = self.vW(h_n)
            Q_tot = torch.cat([Q_tot, Q_n.unsqueeze(1)], dim=1)
            K_tot = torch.cat([K_tot, K_n.unsqueeze(1)], dim=1)
            V_tot = torch.cat([V_tot, V_n.unsqueeze(1)], dim=1)
        
        h_n_attn_out, _ = self.attn(Q_tot.mean(1), K_tot.mean(1), V_tot.mean(1))
        h_final = self.fc(h_n_attn_out[-1])
        return h_final