import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Model(nn.Module):
    def __init__(self, config, n_filters=[2, 8, 16, 32, 64, 128, 256, 512], filter_sizes=[3, 3, 3, 3, 3, 3, 3, 3]):
        super(Model, self).__init__()
        self.postprocess = nn.Conv1d(16, 4, 3, 1, 1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Building the encoder
        for i in range(len(n_filters) - 1):
            self.encoder.append(nn.Conv2d(n_filters[i], n_filters[i + 1], filter_sizes[i], stride=1, padding=1))
            # Initialize weights
            nn.init.uniform_(self.encoder[-1].weight, -1.0 / math.sqrt(n_filters[i]), 1.0 / math.sqrt(n_filters[i]))
            nn.init.constant_(self.encoder[-1].bias, 0)

        # Building the decoder
        n_filters.reverse()
        filter_sizes.reverse()
        for i in range(len(n_filters) - 1):
            self.decoder.append(nn.Conv2d(n_filters[i], n_filters[i + 1], filter_sizes[i], stride=1, padding=1))
            # Initialize weights
            nn.init.uniform_(self.decoder[-1].weight, -1.0 / math.sqrt(n_filters[i]), 1.0 / math.sqrt(n_filters[i]))
            nn.init.constant_(self.decoder[-1].bias, 0)

    def forward(self, x):
        x = rearrange(x, 'b l (s i) -> b i l s', i=2)
        # Encoder
        for layer in self.encoder:
            x = F.tanh(layer(x))

        # Decoder
        for layer in self.decoder:
            x = F.tanh(layer(x))
        # Postprocessor
        x = rearrange(x, 'b i l s -> b l (s i)', i=2)
        x = self.postprocess(x)

        return x