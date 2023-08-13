import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI
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
from filling_the_graphs.lib.nn.models import GRINet

class CSDI_GRIN(nn.Module):
    def __init__(self, config, device):
        super(CSDI_GRIN, self).__init__()
        self.device = device
        self.config = config


        self.target_strategy = config['train']["target_strategy"]
        if 'missing_pattern' in config:
            self.missing_pattern = config['train']['missing_pattern']
        else:
            self.missing_pattern = None

        self.diffmodel = diff_grin(config)

        # parameters for diffusion models
        self.num_steps = config['diffusion']["num_steps"]
        if config['diffusion']["schedule"] == "quad":
            self.beta = np.linspace(
                config['diffusion']["beta_start"] ** 0.5, config['diffusion']["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config['diffusion']["schedule"] == "linear":
            self.beta = np.linspace(
                config['diffusion']["beta_start"], config['diffusion']["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)



    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        if self.config['model']['d_u'] > 0:
            observed_covariates = batch["observed_covariates"].to(self.device).float()
        else:
            observed_covariates = None
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        cut_length = []
        if 'for_pattern_mask' in batch:
            for_pattern_mask = batch["for_pattern_mask"].to(self.device).float()
        else:
            for_pattern_mask = None

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            observed_covariates
        )

    def get_side_info(self, observed_tp, cond_mask, observed_covariates):
        # observed covariates is of shape (B, *, K, L), concat with the side info
        if observed_covariates is not None:
            return observed_covariates    # (B, *, K, L)
        else:
            return None

    def get_randmask(self, observed_mask):
        if self.missing_pattern is None:
            self.missing_pattern = 'random'

        if self.missing_pattern == 'random':  # randomly sample a mask for each batch
            rand_for_mask = torch.rand_like(observed_mask) * observed_mask
            rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
            for i in range(len(observed_mask)):
                sample_ratio = np.random.rand()  # missing ratio
                num_observed = observed_mask[i].sum().item()
                num_masked = round(num_observed * sample_ratio)
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
            cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()

        elif self.missing_pattern == 'block':
            B, K, L = observed_mask.size()  # (B, K, L)
            cond_mask = observed_mask.clone()
            # each batch has a different missing ratio
            for i in range(len(observed_mask)):
                sample_ratio = np.random.rand()  # missing ratio
                expected_num_masked = round(observed_mask[i].sum().item() * sample_ratio)
                cur_num_masked = 0
                # if block size is provided in config, use it
                if 'time_block_size' in self.config['model']:
                    l_block_size = self.config['model']['time_block_size']
                else:
                    l_block_size = np.random.randint(1, L + 1)

                if 'space_block_size' in self.config['model']:
                    k_block_size = self.config['model']['space_block_size']
                else:
                    k_block_size = 1

                while cur_num_masked < expected_num_masked:
                    l_start = np.random.randint(0, L - l_block_size + 1)
                    k_start = np.random.randint(0, K - k_block_size + 1)

                    # if cond_mask is 0, then we don't count it
                    cur_num_masked += cond_mask[i, k_start:k_start + k_block_size,
                                      l_start:l_start + l_block_size].sum().item()

                    # set the mask to 0
                    cond_mask[i, k_start:k_start + k_block_size, l_start:l_start + l_block_size] = 0


        else:
            raise NotImplementedError

        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def calc_loss_valid(
            self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
            self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)  # (B,K,L)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise  # (B,K,L)

        noise = noise.unsqueeze(1)  # (B,1,K,L)
        cond_obs = (cond_mask * observed_data).unsqueeze(1)  # (B,1,K,L)
        noisy_data = noisy_data.unsqueeze(1)  # (B,1,K,L)
        cond_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
        observed_mask = observed_mask.unsqueeze(1)  # (B,1,K,L)
        observed_data = observed_data.unsqueeze(1)  # (B,1,K,L)

        predicted, imputed = self.diffmodel(cond_obs, cond_mask, side_info, noisy_data, t)  # (B,1,K,L)



        if is_train == 1:
            target_mask = observed_mask
        else:
            target_mask = observed_mask - cond_mask

        residual_1 = (noise - predicted) * target_mask
        residual_2 = (observed_data - imputed) * target_mask

        num_eval = target_mask.sum()
        loss = ((residual_1 ** 2).sum() + (residual_2 ** 2).sum()) / (num_eval if num_eval > 0 else 1)
        return loss

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        cond_obs = (cond_mask * observed_data).unsqueeze(1)  # (B,1,K,L)

        observed_data = observed_data.unsqueeze(1)  # (B,1,K,L)
        cond_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)


        for i in range(n_samples):
            # generate noisy observation for unconditional model
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                noisy_data = current_sample
                predicted, _ = self.diffmodel(cond_obs, cond_mask, side_info, noisy_data, torch.tensor([t]).to(self.device))  # (B,1,K,L)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise

            current_sample = current_sample.squeeze(1)
            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples



    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            observed_covariates
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask, observed_covariates)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            observed_covariates
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask, observed_covariates)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0: cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp



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


class diff_grin(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config['diffusion']["num_steps"],
            embedding_dim=config['diffusion']["diffusion_embedding_dim"],
        )

        K = config['model']['K']

        # create a numpy adjacency matrix of K points, using gaussian kernel
        adj = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                adj[i, j] = np.exp(-np.square(i - j) / 2)
        adj = torch.from_numpy(adj).float()
        adj = adj - np.eye(K)

        self.model = GRINet(
                 adj=adj,
                 d_in=1,
                 d_hidden=config['model']['d_hidden'],
                 d_ff=config['model']['d_ff'],
                 ff_dropout=config['model']['ff_dropout'],
                 n_layers=1,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 d_u=config['diffusion']['diffusion_embedding_dim']+config['model']['d_u'],
                 d_emb=0,
                 layer_norm=False,
                 merge='mlp',
                 impute_only_holes=True,
                 d_v=1)

        self.out = nn.Conv1d(4 * config['model']['d_hidden'] + 2, 1, kernel_size=1)




    def forward(self, cond_obs, cond_mask, side_info, noisy_data, diffusion_step):

        B, _, K, L = cond_obs.shape

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # expand diffusion_emb to (B, *, L, K)
        diffusion_emb = diffusion_emb.unsqueeze(-1).unsqueeze(-1) # Now the shape is (B, C, 1, 1)

        # Then, you can use the repeat method to replicate the tensor along the desired dimensions
        diffusion_emb = diffusion_emb.repeat(int(B/diffusion_emb.size(0)), 1, K, L) # Now the shape is (B, C, K, L)

        if side_info is not None:
            side_info = torch.cat([side_info, diffusion_emb], dim=1)
        else:
            side_info = diffusion_emb



        x = cond_obs
        mask = cond_mask.clone()

        # reshape to (B, L, K, -1) for GRINet
        x = x.permute(0, 3, 2, 1)  # (B, L, K, 2)
        side_info = side_info.permute(0, 3, 2, 1)  # (B, L, K, 2)
        mask = mask.permute(0, 3, 2, 1)  # (B, L, K, 2)
        mask = mask.bool()
        noisy_data = noisy_data.permute(0, 3, 2, 1)  # (B, L, K, 1)


        imputation, _, repr = self.model(x, mask=mask, u=side_info, v=noisy_data)  # (B, L, K, *)

        imputation = imputation.permute(0, 3, 2, 1)  # (B, *, K, L)
        repr = repr.permute(0, 3, 2, 1)  # (B, *, K, L)
        repr = repr.reshape(B, -1, K*L)
        y = self.out(repr)  # (B, 1, K*L)
        y = y.reshape(B, 1, K, L)  # (B, 1, K, L)
        return y, imputation

