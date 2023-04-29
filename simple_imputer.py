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
            # training_data[b, (observed_mask[b] - gt_mask[b]).bool()] = torch.mean(training_data[b, gt_mask[b].bool()])

            training_data[b, (observed_mask[b] - gt_mask[b]).bool()] = torch.mean(observed_data[b, target_mask[b].bool()])


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

        for b in range(B):
            for k in range(K):
               training_data[b, :, k] = pd.Series(training_data[b, :, k]).interpolate(method='linear', limit_direction='both').values

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
    from dataset_gsm import get_dataloader
    # from dataset_soilmoisture import get_dataloader
    # from dataset_synthetic import get_dataloader
    # from dataset_pm25 import get_dataloader
    import datetime
    import os

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
            "./save/gsm_mean" + "_" + current_time + "/"
    )

    # foldername = (
    #         "./save/soilmoisture_mean" + "_" + current_time + "/"
    # )

    # foldername = (
    #         "./save/synthetic_mean" + "_" + current_time + "/"
    # )

    # foldername = (
    #         "./save/pm25_mean" + "_" + current_time + "/"
    # )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        missing_data_ratio=0.2,
        training_data_ratio=0.8,
        batch_size=16,
        device=device,
        seed=42
    )

    # # used for pm25 dataset
    # train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    #     16, device=device, validindex=0
    # )

    model = MeanImputer(device=device)
    # model = LinearInterpolationImputer(device=device)

    evaluate(
            model,
            test_loader,
            nsample=10,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername
        )