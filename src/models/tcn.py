"""Causal convolutions: every decision uses only its trailing window."""
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, incoming, outgoing, kernel_size, dilation, dropout):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(incoming, outgoing, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(outgoing, outgoing, kernel_size, dilation=dilation)
        self.residual = nn.Identity() if incoming == outgoing else nn.Conv1d(incoming, outgoing, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.dropout(F.relu(self.conv1(F.pad(x, (self.padding, 0)))))
        y = self.dropout(F.relu(self.conv2(F.pad(y, (self.padding, 0)))))
        return F.relu(y + self.residual(x))


class SmallTCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not config.get('causal') or config['num_classes'] != 2:
            raise ValueError('This experiment requires causal binary classification')
        channels, dilations = config['tcn_channels'], config['dilations']
        if len(channels) != len(dilations) or not channels:
            raise ValueError('One dilation per TCN block is required')
        self.normalization = nn.LayerNorm(config['feature_dim'])
        self.projection = nn.Linear(config['feature_dim'], channels[0])
        blocks = []
        incoming = channels[0]
        for outgoing, dilation in zip(channels, dilations):
            blocks.append(CausalBlock(incoming, outgoing, config['kernel_size'], dilation, config['dropout']))
            incoming = outgoing
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(incoming, 2)

    def temporal_features(self, x):
        return self.blocks(self.projection(self.normalization(x)).transpose(1, 2))

    def forward(self, x):
        return self.classifier(self.temporal_features(x)[:, :, -1])


def build_tcn(config):
    return SmallTCN(config)

