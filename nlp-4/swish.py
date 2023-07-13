import torch
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(input)

# nn.SiLU()即可调用