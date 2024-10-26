from torch import nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Defines a Mixer block, which performs channel mixing and spatial mixing operations
class MixerBlock(nn.Module):
    def __init__(self, dim, num_query, spatial_dim, channel_dim, dropout=0.2):
        super().__init__()

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_query, spatial_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.spatial_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = x + self.channel_mix(x)
        x = x + self.spatial_mix(x)
        x = residual * self.sigmoid(x)
        return x + residual


class DIMixer(nn.Module):
    def __init__(self, dim, depth, num_query):
        super().__init__()

        spatial_dim = num_query
        channel_dim = dim

        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(dim, num_query, spatial_dim, channel_dim) for _ in range(depth)]
        )

    def forward(self, x):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        return x
