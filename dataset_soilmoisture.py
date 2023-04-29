import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch


class SMDataset(Dataset):
    def __init__(self, y, X, observation_mask, gt_mask):

        # numpy, permute to B x K x L
        self.y = np.transpose(y, (0, 2, 1))
        self.observation_mask = np.transpose(observation_mask, (0, 2, 1))
        self.gt_mask = np.transpose(gt_mask, (0, 2, 1))

        # permute (B, L, K, C) to (B, C, K, L)
        self.X = np.transpose(X, (0, 3, 2, 1))
        self.timepoints = np.arange(y.shape[1])


    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        batch = {}
        batch['observed_data'] = self.y[idx, ...]
        batch['observed_mask'] = self.observation_mask[idx, ...]
        batch['timepoints'] = self.timepoints
        batch['gt_mask'] = self.gt_mask[idx, ...]
        batch['observed_covariates'] = self.X[idx, ...]
        return batch


def get_dataloader(missing_data_ratio, training_data_ratio, batch_size, device, seed=0):

    # read data from data/SMAP_Climate_In_Situ_TxSon.csv
    df = pd.read_csv("./data/Insitu_gap_filling_data.csv")
    #df = pd.read_csv("./data/SMAP_Climate_In_Situ_TxSon.csv")
    # number of unique spatial points in df
    K = df["POINTID"].nunique()
    # number of unique dates in df
    L = df["Date"].nunique()


    # reorder the dataframe
    df.sort_values(by=["x", "y"], inplace=True)
    df['x_loc'] = np.unique(df['x'], return_inverse=True)[1]
    df['y_loc'] = np.unique(df['y'], return_inverse=True)[1]
    df['pos'] = df['x_loc'].astype('str') + ',' + df['y_loc'].astype('str')

    def reorder_spatial_index(x, y, xx, yy):
        i = 0
        j = 0
        res = []
        while i < x:
            while j < y:
                for p in range(i, i + xx):
                    for q in range(j, j + yy):
                        res.append([p, q])
                j += yy
            i += xx
            j = 0

        return np.array(res)

    res = reorder_spatial_index(36, 36, 6, 6)
    res = pd.DataFrame(res)
    categories = res.iloc[:, 0].astype('str') + ',' + res.iloc[:, 1].astype('str')
    categories = categories.values
    df['pos'] = pd.Categorical(df['pos'], categories=categories, ordered=True)
    df = df.sort_values(by=['pos', 'Date'])


    # make a numpy array with shape (K, L), that stores the y values
    y = df["SMAP_1km"].values.reshape(K, L)

    # reshape to y to have shape (L, K)
    y = np.transpose(y, (1, 0))

    # number of unique features used to predict y
    C = 6

    # make a numpy array with shape (K, L, C), that stores the x values
    X = df[["prcp", "srad", "tmax", "tmin", "vp", "SMAP_36km"]].values.reshape(K, L, C)

    # reshape to X to have shape (L, K, C)
    X = np.transpose(X, (1, 0, 2))


    T = 108  # make each time series contain T steps
    B = int(y.shape[0] / T)  # how many time series can we split the time period into
    temp = np.vsplit(y[:B * T, :], B)
    temp = [np.hsplit(i, 36) for i in temp]  # split the spatial points into 36 for one field
    temp = [item for sublist in temp for item in sublist]
    temp = np.stack(temp, axis=0)  # each sample is a 6x6 grid with T time steps
    y = np.stack(temp, axis=0)   # shape (B*36, T, 36)


    temp = np.vsplit(X[:B * T, :, :], B)
    temp = [np.hsplit(i, 36) for i in temp]  # split the sptial points into 36 for one field
    temp = [item for sublist in temp for item in sublist]
    temp = np.stack(temp, axis=0)  # each sample is a 6x6 grid with T time steps
    X = np.stack(temp, axis=0)  # shape (B*36, T, 36, C)

    # reset B, T, K, C
    B = y.shape[0]
    T = y.shape[1]
    K = y.shape[2]
    C = X.shape[-1]

    # standardize y using np.nanstd and np.nanmean
    y_std = np.nanstd(y)
    y_mean = np.nanmean(y)
    y_standardized = (y - y_mean) / y_std

    # standardize X using np.nanstd and np.nanmean for each feature c
    X_std = np.nanstd(X, axis=(0, 1, 2))  # shape (C,)
    X_mean = np.nanmean(X, axis=(0, 1, 2))  # shape (C,)
    X_standardized = (X - X_mean) / X_std

    # create a mask for X_standardized, which is True for all non-missing values
    X_standardized_mask = ~np.isnan(X_standardized)

    # fill all missing values in X_standardized with 0
    X_standardized[np.isnan(X_standardized)] = 0

    # concatenate X_standardized and X_standardized_mask to X_standardized
    X_standardized = np.concatenate([X_standardized, X_standardized_mask], axis=-1)  # shape (B, T, K, C + C)


    observation_mask = ~np.isnan(y_standardized)  # True means observed, False means missing

    # randomly mask some data as missing using random state
    rng = np.random.RandomState(seed=seed)
    temp = rng.uniform(size=(B, T, 1)) > missing_data_ratio  # all spatial points are either observed or missing at tone time point
    missing_mask = np.repeat(temp, K, axis=2) & observation_mask  # randomly mask some data as missing


    # copy y_standardized to y_missing_standardized
    y_missing_standardized = y_standardized.copy()

    # replace all missing values and artificially missing values with 0
    y_missing_standardized[~missing_mask] = 0

    # create gt_mask, which additionally mask some data in the remaining non-missing data
    # gt_mask = (rng.uniform(size=y_missing_standardized.shape) < training_data_ratio) & missing_mask  # gt_mask True means used for training
    temp = rng.uniform(size=(B, T, 1)) < training_data_ratio  # all spatial points are either observed or missing at tone time point
    gt_mask = np.repeat(temp, K, axis=2) & missing_mask  # gt_mask True means used for training

    # gt_mask True for training data
    y_training = y_missing_standardized.copy()
    y_training[~gt_mask] = 0


    training_dataset = SMDataset(y_training, X_standardized, gt_mask, gt_mask)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_dataset = SMDataset(y_missing_standardized, X_standardized, missing_mask, gt_mask)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # replace missing values in the real data with 0
    y_standardized[~observation_mask] = 0

    # create test dataset
    # we use the complete data as the test data
    test_dataset = SMDataset(y_standardized, X_standardized, observation_mask, missing_mask)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return training_loader, validation_loader, test_loader, y_std, y_mean


