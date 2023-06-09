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


# def get_tcn(input_size, hidden_size=[16, 32, 64]):
#     return TemporalConvNet(input_size, hidden_size)
#
#
# def get_bigril(input_size, hidden_size=16, n_nodes=1296, n_layers=1, dropout=0.1, order=1):
#     return BiGRIL(input_size=input_size,
#                   hidden_size=hidden_size,
#                   n_nodes=n_nodes,
#                   n_layers=n_layers,
#                   dropout=dropout,
#                   order=order
#                   )
def get_bilstm(channels, hidden_size=64, n_layers=1):
    return BiLSTM(input_size=channels, hidden_size=hidden_size, num_layers=n_layers)




def get_torch_trans(heads=8, layers=1, channels=64, hidden_size=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=hidden_size, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_longformerTS(heads=8, layers=1, channels=64, hidden_size=64, attention_window=27, attention_dilation=1,
                     attention_mode="sliding_chunks"):
    encoder_layer = LongformerTS(d_model=channels, nhead=heads, dim_feedforward=hidden_size, activation="gelu",
                                 attention_window=attention_window, attention_dilation=attention_dilation,
                                 attention_mode=attention_mode)
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


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
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [ResidualBlock(
                side_dim=config["side_dim"],
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                time_layer_config=config["time_layer"],
                spatial_layer_config=config["spatial_layer"]
            )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


# class ResidualBlock_GRIN(nn.Module):
#     def __init__(self, side_dim, channels, diffusion_embedding_dim):
#         super().__init__()
#         self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
#         self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
#         self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
#         self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
#
#         self.spatial_temporal_layer = get_bigril(input_size=channels)
#
#     def forward(self, x, cond_info, diffusion_emb):
#         B, channel, K, L = x.shape
#         x = x.reshape(B, channel, K * L)
#         diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
#         y = x + diffusion_emb
#
#         # use GCRNN
#         y = y.reshape(B, channel, K, L)
#         y = self.spatial_temporal_layer(y)
#         y = y.reshape(B, channel, K * L)
#
#         y = self.mid_projection(y)  # (B,2*channel,K*L)
#         gate, filter = torch.chunk(y, 2, dim=1)  # (B,channel,K*L)
#         y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
#         y = self.output_projection(y)  # (B,2*channel,K*L)
#
#         _, cond_dim, _, _ = cond_info.shape
#         cond_info = cond_info.reshape(B, cond_dim, K * L)
#         cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
#         y = y + cond_info
#
#         residual, skip = torch.chunk(y, 2, dim=1)
#         x = x.reshape([B, channel, K, L])
#         residual = residual.reshape([B, channel, K, L])
#         skip = skip.reshape([B, channel, K, L])
#         return (x + residual) / math.sqrt(2.0), skip


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, time_layer_config, spatial_layer_config):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer_type = time_layer_config['type']
        self.spatial_layer_type = spatial_layer_config['type']

        # time layer configuraton
        if self.time_layer_type == 'transformer':
            self.time_layer = get_torch_trans(heads=time_layer_config['nheads'], layers=1, channels=channels,
                                              hidden_size=time_layer_config['hidden_size'])
        elif self.time_layer_type == 'longformer':
            self.time_layer = get_longformerTS(heads=spatial_layer_config['nheads'], layers=1, channels=channels,
                                               hidden_size=time_layer_config['hidden_size'],
                                               attention_window=time_layer_config['attention_window'])
        elif self.time_layer_type == 'bilstm':
            self.time_layer = get_bilstm(channels=channels,
                                         hidden_size=time_layer_config['hidden_size']
                                         )
        else:
            self.time_layer = None

        # spatial layer configuration
        if self.spatial_layer_type == 'transformer':
            self.feature_layer = get_torch_trans(heads=spatial_layer_config['nheads'], layers=1, channels=channels,
                                                 hidden_size=spatial_layer_config['hidden_size'])
        elif self.spatial_layer_type == 'diffconv':
            # read adj matrix from file
            adj = np.load(spatial_layer_config['adj'])
            # convert to torch tensor
            adj = torch.from_numpy(adj).float()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            adj = adj.to(device)
            self.feature_layer = get_spatial_diffusion_conv(channels=channels,
                                                            hidden_size=spatial_layer_config['hidden_size'],
                                                            order=spatial_layer_config['order'], adj=adj,
                                                            include_self=True)
        else:
            self.feature_layer = None

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

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        if self.time_layer is not None:
            y = self.forward_time(y, base_shape)
        if self.feature_layer is not None:
            y = self.forward_feature(y, base_shape)

        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)  # (B,channel,K*L)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)  # (B,2*channel,K*L)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip