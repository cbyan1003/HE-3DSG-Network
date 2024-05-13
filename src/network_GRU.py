import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network_GNN import GraphEdgeAttenNetworkLayers
from config import Config

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  
        self.grucell = nn.GRUCell(input_size, hidden_size)
    def forward(self, x, h):
        return self.grucell(x, h) # [5,512]  [10,5,512]
        