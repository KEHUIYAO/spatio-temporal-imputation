from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd

def generate_tmax_data():
    data = pd.read_csv('./data/Tmax.csv')
    y = data.loc[:, '3804': '94930'].values
    # if the values is less than -9998, make it as missing value
    y = y.astype(float)
    y[y < -9998] = np.nan

    # y is a matrix of size (L, K), let's transpose it to (K, L)
    y = y.T  # (328, 1461)
    y = y[:, :1400]  # (328, 1400)

    y_mean = np.nanmean(y)
    y_std = np.nanstd(y)
    y = (y - y_mean) / y_std

    # reshape it to (328 x 14, 1, 100)
    y = y.reshape(328*14, 1, 100)  # B = 328 x 14 = 4592, K=1, L=100
    return y, y_mean, y_std

if __name__ == '__main__':
    from dataset import get_dataloader
    y, y_mean, y_std = generate_tmax_data()
    print(y.shape)

    res = get_dataloader(y, y_mean, y_std)