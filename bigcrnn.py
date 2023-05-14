import numpy as np
import torch
import torch.nn as nn
from nn.layers.gcn import GCN
class BiGCRNN(nn.Module):
    def __init__(self, target_size, covariate_size, config, device, adj=None, hidden_size=64, num_layers=1, dropout=0.0):
        super(BiGCRNN, self).__init__()
        self.config = config
        self.device = device
        self.target_strategy = config['model']['target_strategy']
        self.covariate_size = covariate_size
        if 'missing_pattern' in config['model']:
            self.missing_pattern = config['model']['missing_pattern']
        else:
            self.missing_pattern = None

        self.hidden_size = hidden_size
        input_size = covariate_size + 2  # 2 for observed data and mask

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, dropout=dropout)
        self.out = nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=1)

        self.gcn = GCN(2*hidden_size, hidden_size, hidden_size, dropout=dropout)
        if adj is not None:
            self.adj = torch.from_numpy(adj).float().to(device)
        else:
            # fully connected adj matrix
            self.adj = torch.ones((target_size, target_size)).float().to(device)
            # diagnoal is zero
            self.adj = self.adj - torch.eye(target_size).float().to(device)

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

        # observed_data: B x K x L
        # observed_mask: B x K x L
        # observed covariates: B x C x K x L
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != 'random':
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        B, K, L = observed_data.shape
        C = observed_covariates.shape[1] if observed_covariates is not None else 0
        target_mask = observed_mask - cond_mask
        non_missing_data = (observed_data * cond_mask).unsqueeze(1)
        if observed_covariates is not None:
            total_input = torch.cat([non_missing_data, cond_mask.unsqueeze(1), observed_covariates], dim=1)
        else:
            total_input = torch.cat([non_missing_data, cond_mask.unsqueeze(1)], dim=1)  # B x (C+2) x K x L

        output = self.forward_time(total_input)
        output = self.forward_space(output, self.adj)  # B x K x H x L

        output = output.reshape(B*K, self.hidden_size, L)  # B*K x H x L

        output = self.out(output)  # B*K x 1 x L

        # permute back to the original order BxKxL
        predicted = output.reshape(B, K, L)

        residual = (observed_data - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def forward_time(self, x):
        # x: B x C x K x L
        B, C, K, L = x.shape
        total_input = x.permute(3, 0, 2, 1).reshape(L, B*K, -1)
        predicted = self.rnn(total_input)[0]  # L x B*K x 2H
        # permute to the input for the convolutional layer
        predicted = predicted.permute(1, 2, 0)  # B*K x 2H x L

        # reshape to B x 2H x K x L
        predicted = predicted.reshape(B, K, 2*self.hidden_size, L)  # B x K x 2H x L
        return predicted

    def forward_space(self, x, adj):
        # x: B x C x K x L
        B, K, input_size, L = x.shape
        # reshape input to be B*L x K x input_size
        total_input = x.permute(0, 3, 1, 2).reshape(B*L, K, input_size)
        output = self.gcn(total_input, self.adj)  # B*L x K x H
        # reshape output to be B x L x K x H
        output = output.reshape(B, L, K, -1).permute(0, 2, 3, 1)  # B x K x H x L
        return output


    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        if self.covariate_size > 0 and 'observed_covariates' in batch:
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

    def get_randmask(self, observed_mask):
        if self.missing_pattern is None:
            self.missing_pattern = 'random'

        if self.missing_pattern == 'random':
            rand_for_mask = torch.rand_like(observed_mask) * observed_mask
            rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
            for i in range(len(observed_mask)):
                sample_ratio = np.random.rand()  # missing ratio
                num_observed = observed_mask[i].sum().item()
                num_masked = round(num_observed * sample_ratio)
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
            cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        elif self.missing_pattern == 'space_block':  #  all spatial locations at one specific time point are either observed or masked
            B, K, L = observed_mask.size()  # (B, K, L)
            cond_mask = observed_mask.clone()
            # each batch has a different missing ratio
            for i in range(len(observed_mask)):
                # randomly generate a number from 0 to 1
                sample_ratio = np.random.rand()  # missing ratio
                temp = torch.rand(size=(1, L), device=self.device) < sample_ratio
                # repeat the mask for all spatial points
                cond_mask[i, :, :] = cond_mask[i, :, :] * temp.expand(K, L)
        elif self.missing_pattern == 'time_block':  #  all time points at one specific spatial location are either observed or masked
            B, K, L = observed_mask.size()  # (B, K, L)
            cond_mask = observed_mask.clone()
            # each batch has a different missing ratio
            for i in range(len(observed_mask)):
                # randomly generate a number from 0 to 1
                sample_ratio = np.random.rand()
                temp = torch.rand(size=(K, 1), device=self.device) < sample_ratio
                # repeat the mask for all spatial points
                cond_mask[i, :, :] = cond_mask[i, :, :] * temp.expand(K, L)
        elif self.missing_pattern == 'block':
            B, K, L = observed_mask.size()  # (B, K, L)
            cond_mask = observed_mask.clone()
            # each batch has a different missing ratio
            for i in range(len(observed_mask)):
                sample_ratio = np.random.rand()  # missing ratio
                expected_num_masked = round(K * L * sample_ratio)
                cur_num_masked = 0
                # if block size is provided in config, use it
                if 'time_block_size' in self.config['model']:
                    l_block_size = self.config['model']['time_block_size']
                else:
                    l_block_size = np.random.randint(1, L + 1)

                if 'space_block_size' in self.config['model']:
                    k_block_size = self.config['model']['space_block_size']
                else:
                    k_block_size = np.random.randint(1, K + 1)

                while cur_num_masked < expected_num_masked:
                    l_start = np.random.randint(0, L - l_block_size + 1)
                    k_start = np.random.randint(0, K - k_block_size + 1)
                    cond_mask[i, k_start:k_start + k_block_size, l_start:l_start + l_block_size] = 0
                    cur_num_masked += l_block_size * k_block_size

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
        # observed_data: B x K x L
        # observed_mask: B x K x L
        # observed covariates: B x C x K x L
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
            B, K, L = observed_data.shape
            C = observed_covariates.shape[1] if observed_covariates is not None else 0
            target_mask = observed_mask - cond_mask
            non_missing_data = (observed_data * cond_mask).unsqueeze(1)
            if observed_covariates is not None:
                total_input = torch.cat([non_missing_data, cond_mask.unsqueeze(1), observed_covariates], dim=1)
            else:
                total_input = torch.cat([non_missing_data, cond_mask.unsqueeze(1)], dim=1)


            output = self.forward_time(total_input)
            output = self.forward_space(output, adj=self.adj)  # B x K x H x L
            output = output.reshape(B * K, self.hidden_size, L)  # B*K x H x L

            output = self.out(output)  # B*K x 1 x L

            # permute back to the original order BxKxL
            predicted = output.reshape(B, 1, K, L)  # B x 1 x K x L

            return predicted, observed_data, target_mask, observed_mask, observed_tp


if __name__ == '__main__':
    pass
