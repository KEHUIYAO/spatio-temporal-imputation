import numpy as np
import pandas as pd

def generate_heating_data():

    data = pd.read_csv('./data/tsHeating_complete.csv')

    y = data['x'].values

    y_mean = np.nanmean(y)
    y_std = np.nanstd(y)

    y = (y - y_mean) / y_std


    y = y[:606800]
    y = y.reshape(6068, 1, 100)

    observation_mask = ~np.isnan(y)

    print("y_mean: ", y_mean)
    print("y_std: ", y_std)

    return y, y_mean, y_std, observation_mask

if __name__ == '__main__':
    from dataloader import get_dataloader
    y, y_mean, y_std, observation_mask = generate_heating_data()
    print(y.shape)

    res = get_dataloader(y, y_mean, y_std, observation_mask=observation_mask)