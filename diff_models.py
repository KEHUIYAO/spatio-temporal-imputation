import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nn.layers.gril import BiGRIL
from nn.layers.tcn import TemporalConvNet
from nn.layers.longformer import LongformerTS
from nn.layers.spatial_conv import SpatialDiffusionConv
from nn.layers.bilstm import BiLSTM

def get_bilstm(channels, hidden_size=64, n_layers=1):
    return BiLSTM(input_size=channels, hidden_size=hidden_size, num_layers=n_layers)


def get_spatial_diffusion_conv(channels, hidden_size, order, adj, include_self=True):
    return SpatialDiffusionConv(
        c_in=channels,
        c_out=hidden_size,
        adj=adj,
        order=order,
        include_self=include_self
    )


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.diffusion_projection = nn.Linear(config['diffusion_embedding_dim'], self.channels)
        self.output_projection = Conv1d_with_init(self.channels, 1, 1)

        self.time_layer_type = config["time_layer"]['type']
        self.spatial_layer_type = config["spatial_layer"]["type"]

        if self.time_layer_type == "bilstm":
            print('use bilstm for temporal modeling')
            self.time_layer = get_bilstm(channels=self.channels,
                                         hidden_size=config["time_layer"]['hidden_size']
                                         )
        else:
            self.time_layer = None

        if self.spatial_layer_type == "diffconv":
            print('use diffconv for spatial modeling')
            # read adj matrix from file
            adj = np.load(config['spatial_layer']['adj'])
            # convert to torch tensor
            adj = torch.from_numpy(adj).float()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            adj = adj.to(device)
            self.feature_layer = get_spatial_diffusion_conv(channels=self.channels,
                                                            hidden_size=config['spatial_layer']['hidden_size'],
                                                            order=config['spatial_layer']['order'],
                                                            adj=adj,
                                                            include_self=True)
        else:
            self.feature_layer = None


    def forward(self, x, cond_info, diffusion_step):
        # concatenate the cond_info to the input
        x = torch.cat([x, cond_info], dim=1)
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)


        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        if self.time_layer is not None:
            y = self.forward_time(y, base_shape)
        if self.feature_layer is not None:
            y = self.forward_feature(y, base_shape)

        y = self.output_projection(y)  # (B,1,K*L)
        y = y.reshape(B, K, L)
        return y

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y
