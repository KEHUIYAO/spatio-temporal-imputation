import numpy as np
import pandas as pd

def generate_heating_data():
    data_missing = pd.read_csv('./data/tsHeating_missing.csv')
    data_complete = pd.read_csv('./data/tsHeating_complete.csv')

    y_missing = data_missing['x'].values
    y = data_complete['x'].values
    y_mean = np.nanmean(y_missing)
    y_std = np.nanstd(y_missing)

    y = (y - y_mean) / y_std

    y_missing = y_missing[:606800]
    y = y[:606800]
    y_missing = y_missing.reshape(6068, 1, 100)
    y = y.reshape(6068, 1, 100)

    observation_mask = ~np.isnan(y)
    missing_mask = ~np.isnan(y_missing) & observation_mask

    print("y_mean: ", y_mean)
    print("y_std: ", y_std)

    return y, y_mean, y_std, observation_mask, missing_mask

if __name__ == '__main__':
    from dataloader import get_dataloader
    y, y_mean, y_std, observation_mask, missing_mask = generate_heating_data()
    print(y.shape)
    print(missing_mask.shape)

    res = get_dataloader(y, y_mean, y_std, observation_mask=observation_mask, missing_mask=missing_mask)