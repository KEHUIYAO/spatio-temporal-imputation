from filling_the_graphs.lib.nn.models import GRINet
import torch
import torch.nn as nn
import numpy as np

class GRIN(nn.Module):
    def __init__(self,
                 adj,
                 d_in,
                 d_hidden,
                 d_ff,
                 ff_dropout,
                 config,
                 device,
                 n_layers=1,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 d_u=0,
                 d_emb=0,
                 layer_norm=False,
                 merge='mlp',
                 impute_only_holes=True):
        super(GRIN, self).__init__()
        self.config = config
        self.device = device
        self.target_strategy = config['model']['target_strategy']
        if 'missing_pattern' in config['model']:
            self.missing_pattern = config['model']['missing_pattern']
        else:
            self.missing_pattern = None


        self.model = GRINet(adj,
                 d_in,
                 d_hidden,
                 d_ff,
                 ff_dropout,
                 n_layers=1,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 d_u=0,
                 d_emb=0,
                 layer_norm=False,
                 merge='mlp',
                 impute_only_holes=True)


    def forward(self, batch, is_train=1):
        # observed_data: B x K x L
        # observed_mask: B x K x L
        # observed covariates: B x C x K x L
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length
        ) = self.process_data(batch)

        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != 'random':
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        B, K, L = observed_data.shape
        if is_train == 1:
            target_mask = observed_mask
        else:
            target_mask = observed_mask - cond_mask
        non_missing_data = (observed_data * cond_mask).unsqueeze(1)  # B, 1, K, L
        cond_mask = cond_mask.unsqueeze(1)  # B, 1, K, L

        # permute to B, L, K, 1
        non_missing_data = non_missing_data.permute(0, 3, 2, 1)  # B, L, K, 1
        cond_mask = cond_mask.permute(0, 3, 2, 1)  # B, L, K, 1

        # convert float to bool
        cond_mask = cond_mask.bool()


        predicted = self.model(non_missing_data, cond_mask)  # B, L, K, 1

        # permute back to B, K, L
        predicted = predicted.permute(0, 3, 2, 1)  # B, 1, K, L

        # squeeze to B, K, L
        predicted = predicted.squeeze(1)  # B, K, L


        residual = (observed_data - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)


        return loss





    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
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
            cut_length
        )

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
                    l_block_size = np.random.randint(1, L+1)

                if 'space_block_size' in self.config['model']:
                    k_block_size = self.config['model']['space_block_size']
                else:
                    k_block_size = 1

                while cur_num_masked < expected_num_masked:
                    l_start = np.random.randint(0, L-l_block_size+1)
                    k_start = np.random.randint(0, K-k_block_size+1)

                    # if cond_mask is 0, then we don't count it
                    cur_num_masked += cond_mask[i, k_start:k_start+k_block_size, l_start:l_start+l_block_size].sum().item()

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


    def evaluate(self, batch, nsample=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            B, K, L = observed_data.shape

            non_missing_data = (observed_data * cond_mask).unsqueeze(1)  # B, 1, K, L
            cond_mask = cond_mask.unsqueeze(1)  # B, 1, K, L

            # permute to B, L, K, 1
            non_missing_data = non_missing_data.permute(0, 3, 2, 1)  # B, L, K, 1
            cond_mask = cond_mask.permute(0, 3, 2, 1)  # B, L, K, 1

            # convert float to bool
            cond_mask = cond_mask.bool()


            predicted = self.model(non_missing_data, cond_mask)  # B, L, K, 1

            # permute to (B, 1, K, L)
            predicted = predicted.permute(0, 3, 2, 1)  # B, 1, K, L

            return predicted, observed_data, target_mask, observed_mask, observed_tp