import numpy as np
import torch
import torch.nn as nn

class BiRNN(nn.Module):
    def __init__(self, covariate_size, config, device, hidden_size=64, num_layers=1, dropout=0.0, task_type='mit'):
        super(BiRNN, self).__init__()
        self.device = device
        self.target_strategy = config['model']['target_strategy']
        self.covariate_size = covariate_size
        if 'missing_pattern' in config['model']:
            self.missing_pattern = config['model']['missing_pattern']
        else:
            self.missing_pattern = None
        self.task_type = task_type
        hidden_size = hidden_size
        num_layers = num_layers
        input_size = covariate_size + 2  # 2 for observed data and mask

        if task_type == 'mit':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, dropout=dropout)
            self.out = nn.Conv1d(in_channels=2*hidden_size, out_channels=1, kernel_size=1)
        elif task_type == 'ort':
            self.rnn_forward = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
            self.rnn_backward = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
            self.out = nn.Conv1d(in_channels=2*hidden_size, out_channels=1, kernel_size=1)



    def forward(self, batch, is_train=1):
        if self.task_type == 'mit':  # masked imputation training
            return self.calc_loss_mit(batch, is_train)
        elif self.task_type == 'ort':  # observed data reconstruction training
            return self.calc_loss_ort(batch, is_train)


    def calc_loss_mit(self, batch, is_train=1):
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
        target_mask= observed_mask - cond_mask
        non_missing_data = (observed_data * cond_mask).unsqueeze(1)
        if observed_covariates is not None:
            total_input = torch.cat([non_missing_data, cond_mask.unsqueeze(1), observed_covariates], dim=1)
        else:
            total_input = torch.cat([non_missing_data, cond_mask.unsqueeze(1)], dim=1)

        total_input = total_input.permute(3, 0, 2, 1).reshape(L, B*K, -1)

        predicted = self.rnn(total_input)[0]  # L x B*K x 2H
        # permute to the input for the convolutional layer
        predicted = predicted.permute(1, 2, 0)  # B*K x 2H x L
        predicted = self.out(predicted)  # B*K x 1 x L
        # permute back to the original order BxKxL
        predicted = predicted.reshape(B, K, L)

        residual = (observed_data - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def calc_loss_ort(self, batch, is_train=1):
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

        B, K, L = observed_data.shape
        C = observed_covariates.shape[1] if observed_covariates is not None else 0

        if is_train == 1:
            target_mask = observed_mask
            cond_mask = observed_mask
            non_missing_data = observed_data.unsqueeze(1)
        else:
            cond_mask = gt_mask
            target_mask = observed_mask - gt_mask
            non_missing_data = (observed_data * gt_mask).unsqueeze(1)




        if observed_covariates is not None:
            total_input = torch.cat([non_missing_data, cond_mask.unsqueeze(1), observed_covariates], dim=1)
        else:
            total_input = torch.cat([non_missing_data, cond_mask.unsqueeze(1)], dim=1)

        total_input = total_input.permute(3, 0, 2, 1).reshape(L, B * K, -1)

        forward_hidden = self.rnn_forward(total_input)[0]  # L x B*K x H
        backward_hidden = self.rnn_backward(torch.flip(total_input, dims=[0]))[0]  # L x B*K x H
        backward_hidden = torch.flip(backward_hidden, dims=[0])


        forward_hidden = forward_hidden[:-1, :, :]  # L-1 x B*K x H
        backward_hidden = backward_hidden[1:, :, :]  # L-1 x B*K x H

        forward_hidden = forward_hidden.permute(1, 2, 0)  # B*K x H x L-1
        backward_hidden = backward_hidden.permute(1, 2, 0)  # B*K x H x L-1

        hidden = torch.cat([
                           torch.cat([backward_hidden[:, :, 0:1], backward_hidden[:, :, 0:1]], dim=1),
                           torch.cat([forward_hidden[:, :, :-1], backward_hidden[:, :, 1:]], dim=1),
                           torch.cat([forward_hidden[:, :, -1:], forward_hidden[:, :, -1:]], dim=1)
                               ], dim=-1)  # B*K x 2H x L

        output = self.out(hidden)  # B*K x 1 x L

        # forward_output = self.out_forward(forward_hidden.permute(1, 2, 0))  # B*K x 1 x L-1
        # backward_output = self.out_backward(backward_hidden.permute(1, 2, 0))  # B*K x 1 x L-1

        # forward_output corresponds to t=2:L
        # backward_output corresponds to t=1:L-1
        # so t=1, we use backward_output[:, :, 0:1], which is B*K x 1 x 1
        # t=L, we use forward_output[:, :, -1:], which is B*K x 1 x 1
        # t=2...L-1, we use both the average of forward_output and backward_output
        # output = torch.cat([
        #                     backward_output[:, :, 0:1],
        #                    (forward_output[:, :, :-1] + backward_output[:, :, 1:]) / 2,
        #                     forward_output[:, :, -1:]
        #                        ], dim=-1)  # B*K x 1 x L

        # output = torch.cat([
        #                     backward_output[:, :, 0:1],
        #                     forward_output
        #                        ], dim=-1)  # B*K x 1 x L

        # output = torch.cat([
        #     backward_output,
        #     forward_output[:, :, -1:]
        # ], dim=-1)  # B*K x 1 x L

        # permute back to the original order BxKxL
        predicted = output.reshape(B, K, L)
        residual = (observed_data - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss



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
                # randomly generate a number from 0 to 1
                l_block_size = np.random.randint(0, L + 1)
                k_block_size = np.random.randint(0, K + 1)
                l_start = np.random.randint(0, L - l_block_size + 1)
                k_start = np.random.randint(0, K - k_block_size + 1)
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

            total_input = total_input.permute(3, 0, 2, 1).reshape(L, B * K, C+2)


            if self.task_type == 'mit':
                predicted = self.rnn(total_input)[0]  # L x B*K x 2H
                # permute to the input for the convolutional layer
                predicted = predicted.permute(1, 2, 0)  # B*K x 2H x L
                predicted = self.out(predicted)  # B*K x 1 x L
                # permute back to the original order BxKxL
                predicted = predicted.reshape(B, 1, K, L)  # B x 1 x K x L
            elif self.task_type == 'ort':
                forward_hidden = self.rnn_forward(total_input)[0]  # L x B*K x H
                backward_hidden = self.rnn_backward(torch.flip(total_input, dims=[0]))[0]  # L x B*K x H
                backward_hidden = torch.flip(backward_hidden, dims=[0])

                forward_hidden = forward_hidden[:-1, :, :]  # L-1 x B*K x H
                backward_hidden = backward_hidden[1:, :, :]  # L-1 x B*K x H

                forward_hidden = forward_hidden.permute(1, 2, 0)  # B*K x H x L-1
                backward_hidden = backward_hidden.permute(1, 2, 0)  # B*K x H x L-1

                hidden = torch.cat([
                    torch.cat([backward_hidden[:, :, 0:1], backward_hidden[:, :, 0:1]], dim=1),
                    torch.cat([forward_hidden[:, :, :-1], backward_hidden[:, :, 1:]], dim=1),
                    torch.cat([forward_hidden[:, :, -1:], forward_hidden[:, :, -1:]], dim=1)
                ], dim=-1)  # B*K x 2H x L

                output = self.out(hidden)  # B*K x 1 x L

                # permute back to the original order BxKxL
                predicted = output.reshape(B, 1, K, L)

            return predicted, observed_data, target_mask, observed_mask, observed_tp

