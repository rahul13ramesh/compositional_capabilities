import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class AutoLstm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lstm = nn.LSTM(config.n_embd, config.n_embd,
                            num_layers=config.n_layer,
                            batch_first=True, 
                            bias=True)
        self.fc = nn.Linear(config.n_embd, config.vocab_size)
        self.hidden = None
        self.use_hidden=False

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param.data, gain=1.0)

    def forward(self, inp):
        x_embd = self.wte(inp)

        if self.use_hidden:
            lstm_out, (hidden, cell) = self.lstm(x_embd, self.hidden)
            self.hidden = (hidden, cell)
        else:
            lstm_out, (hidden, cell) = self.lstm(x_embd, None)

        logits = self.fc(lstm_out)

        return logits 
