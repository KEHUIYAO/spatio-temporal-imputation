import numpy as np
import pandas as pd
import torch
from utils import train, evaluate


class MeanImputer():
    def __init__(self, device):
        self.device = device

    def eval(self):
        pass

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        return (
            observed_data,
            observed_mask,
            gt_mask,
            observed_tp
        )

    def evaluate(self, batch, nsample=10):
        # observed_data: B x K x L
        # observed_mask: B x K x L
        # observed covariates: B x C x K x L
        (
            observed_data,
            observed_mask,
            gt_mask,
            observed_tp
        ) = self.process_data(batch)

        target_mask = observed_mask - gt_mask

        training_data = observed_data * gt_mask  # B x K x L
        B = training_data.shape[0]

        for b in range(B):
            training_data[b, (observed_mask[b] - gt_mask[b]).bool()] = torch.mean(training_data[b, gt_mask[b].bool()])

            # training_data[b, (observed_mask[b] - gt_mask[b]).bool()] = torch.mean(observed_data[b, target_mask[b].bool()])


            # it is possible that after imputation, the training data still contains nan values, this is becuase the data you use to impute contains all nan. Next, we replace the remaining nan values with 0
        training_data[torch.isnan(training_data)] = 0

        training_data = training_data.unsqueeze(1)  # B x 1 x K x L

        return training_data, observed_data, target_mask, observed_mask, observed_tp


class LinearInterpolationImputer():
    def __init__(self, device):
        self.device = device

    def eval(self):
        pass

    def process_data(self, batch):
        observed_data = batch["observed_data"].float().detach().numpy()
        observed_mask = batch["observed_mask"].float().detach().numpy()
        gt_mask = batch["gt_mask"].float().detach().numpy()
        observed_tp = batch["timepoints"].to(self.device).float()

        return (
            observed_data,
            observed_mask,
            gt_mask,
            observed_tp
        )

    def evaluate(self, batch, nsample=10):
        # observed_data: B x K x L
        # observed_mask: B x K x L
        # observed covariates: B x C x K x L
        (
            observed_data,
            observed_mask,
            gt_mask,
            observed_tp
        ) = self.process_data(batch)

        target_mask = observed_mask - gt_mask

        training_data = observed_data * gt_mask  # B x K x L
        B = training_data.shape[0]
        K = training_data.shape[1]

        training_data[gt_mask==0] = np.nan

        for b in range(B):
            for k in range(K):
                # set the unobserved data to nan, ~gtmask is the unobserved data
                training_data[b, k, :] = pd.Series(training_data[b, k, :]).interpolate(method='linear', limit_direction='both').values

        # it is possible that after imputation, the training data still contains nan values, this is becuase the data you use to impute contains all nan. Next, we replace the remaining nan values with 0
        training_data[np.isnan(training_data)] = 0

        # convert to torch tensor
        training_data = torch.from_numpy(training_data).float().to(self.device)
        training_data = training_data.unsqueeze(1)  # B x 1 x K x L
        observed_data = torch.from_numpy(observed_data).float().to(self.device)
        observed_mask = torch.from_numpy(observed_mask).float().to(self.device)
        target_mask = torch.from_numpy(target_mask).float().to(self.device)

        return training_data, observed_data, target_mask, observed_mask, observed_tp

class KrigingImputer():
    def __init__(self, device, mu, covariance_matrix):
        self.device = device
        self.covariance_matrix = covariance_matrix
        self.mu = mu
        self.covariance_matrix = covariance_matrix

    def eval(self):
        pass

    def conditional_mean(self, mu, sigma, y_given, indices_x, indices_y):
        mu_x = mu[indices_x]
        mu_y = mu[indices_y]

        sigma_x_y = sigma[np.ix_(indices_x, indices_y)]
        sigma_y = sigma[np.ix_(indices_y, indices_y)]

        conditional_mean_x = mu_x + np.matmul(sigma_x_y, np.linalg.solve(sigma_y, y_given - mu_y))
        return conditional_mean_x

    def process_data(self, batch):
        observed_data = batch["observed_data"].float().detach().numpy()
        observed_mask = batch["observed_mask"].float().detach().numpy()
        gt_mask = batch["gt_mask"].float().detach().numpy()
        observed_tp = batch["timepoints"].to(self.device).float()

        return (
            observed_data,
            observed_mask,
            gt_mask,
            observed_tp
        )

    def evaluate(self, batch, nsample=10):
        # observed_data: B x K x L
        # observed_mask: B x K x L
        # observed covariates: B x C x K x L
        (
            observed_data,
            observed_mask,
            gt_mask,
            observed_tp
        ) = self.process_data(batch)

        target_mask = observed_mask - gt_mask

        training_data = observed_data * gt_mask  # B x K x L
        B = training_data.shape[0]
        K = training_data.shape[1]

        training_data[gt_mask==0] = np.nan

        for i in range(B):
            y_given = training_data[i, gt_mask[i]==1]
            indices_x = np.where(target_mask[i].reshape(-1)==1)[0]  # the indices of the unobserved data
            indices_y = np.where(gt_mask[i].reshape(-1)==1)[0]  # the indices of the observed data
            x_pred = self.conditional_mean(self.mu, self.covariance_matrix, y_given, indices_x, indices_y)
            training_data[i, target_mask[i]==1] = x_pred

        # it is possible that after imputation, the training data still contains nan values, this is becuase the data you use to impute contains all nan. Next, we replace the remaining nan values with 0
        training_data[np.isnan(training_data)] = 0

        # convert to torch tensor
        training_data = torch.from_numpy(training_data).float().to(self.device)
        training_data = training_data.unsqueeze(1)  # B x 1 x K x L
        observed_data = torch.from_numpy(observed_data).float().to(self.device)
        observed_mask = torch.from_numpy(observed_mask).float().to(self.device)
        target_mask = torch.from_numpy(target_mask).float().to(self.device)

        return training_data, observed_data, target_mask, observed_mask, observed_tp


if __name__ == '__main__':
    pass