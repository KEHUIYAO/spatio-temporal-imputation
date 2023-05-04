import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


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




def reorder_spatial_index(x, y, xx, yy):
    i = 0
    j = 0
    res = []
    while i < x:
        while j < y:
            for p in range(i, i+xx):
                for q in range(j, j+yy):
                    res.append([p, q])
            j += yy
        i += xx
        j = 0

    return np.array(res)

def process_raw_sm_data(index):
    time_start = '2017-01-01'
    time_end = '2019-12-31'
    time_period = list(pd.date_range(start=time_start, end=time_end).astype(str))
    unique_LC = pd.DataFrame({'LC_36km': np.array([1, 8, 9, 10, 5, 4, 2, 7, 3, 6, 16, 12, 14]),
                              'LC_1km': np.array([1, 8, 9, 10, 5, 4, 2, 7, 3, 6, 16, 12, 14])}
                             )
    one_hot_encoder = OneHotEncoder().fit(unique_LC)

    file_smap_36km = '/home/kehuiyao/google_drive/SoilMoisture_Kehui/smap_36km/smap_36km_index_' + str(index) + '.csv'

    df_smap_36km = pd.read_csv(file_smap_36km)
    df_smap_36km_columns = list(df_smap_36km.columns)

    temp = pd.concat([df_smap_36km.loc[:, ['id', 'index']],
                      pd.DataFrame(np.nan, columns=time_period, index=np.arange(df_smap_36km.shape[0]))], axis=1)

    for day in time_period:
        if day in df_smap_36km_columns:
            temp[day] = df_smap_36km[day]

    df_smap_36km = temp
    df_smap_36km = df_smap_36km.sort_values(['index', 'id'])
    # print(df_smap_36km.shape)

    file_static = '/home/kehuiyao/google_drive/SoilMoisture_Kehui/smap_36km/smap_36km_index_' + str(index) + '.csv'
    df_static = pd.read_csv(file_static)
    df_static = pd.concat([df_static.loc[:, ['id', 'index']], df_static.loc[:, 'elevation':'LC_36km']], axis=1)
    df_static = df_static.sort_values(['index', 'id'])

    # print(df_static.shape)

    file_smap_1km = '/home/kehuiyao/google_drive/SoilMoisture_Kehui/smap_1km/smap_1km_index_' + str(index) + '.csv'

    df_smap_1km = pd.read_csv(file_smap_1km)
    df_smap_1km_columns = list(df_smap_1km.columns)

    temp = pd.concat([df_smap_1km.loc[:, ['id', 'index']],
                      pd.DataFrame(np.nan, columns=time_period, index=np.arange(df_smap_1km.shape[0]))], axis=1)

    for day in time_period:
        if day in df_smap_1km_columns:
            temp[day] = df_smap_1km[day]
    df_smap_1km = temp

    df_smap_1km = df_smap_1km.sort_values(['index', 'id'])
    # print(df_smap_1km.shape)

    file_dpt = '/home/kehuiyao/google_drive/SoilMoisture_Kehui/Climate/dpt_grid_' + str(index) + '.csv'
    df_dpt = pd.read_csv(file_dpt)

    df_dpt_columns = list(df_dpt.columns)
    df_dpt_columns[0] = 'Date'
    df_dpt_columns[3] = 'dpt'
    df_dpt.columns = df_dpt_columns
    df_dpt['Date'] = pd.to_datetime(df_dpt['Date'].apply(lambda x: x[:8])).astype(str)
    df_dpt = df_dpt[['id', 'index', 'Date', 'dpt']]
    df_dpt = df_dpt.pivot(index=['id', 'index'], columns='Date', values='dpt').reset_index()
    df_dpt = df_dpt.rename_axis(None, axis=1)

    df_dpt = df_dpt.sort_values(['index', 'id'])
    # print(df_dpt.shape)

    file_precip = '/home/kehuiyao/google_drive/SoilMoisture_Kehui/Climate/precip_grid_' + str(index) + '.csv'
    df_precip = pd.read_csv(file_precip)

    df_precip_columns = list(df_precip.columns)
    df_precip_columns[0] = 'Date'
    df_precip_columns[3] = 'precip'
    df_precip.columns = df_precip_columns
    df_precip['Date'] = pd.to_datetime(df_precip['Date'].apply(lambda x: x[:8])).astype(str)
    df_precip = df_precip[['id', 'index', 'Date', 'precip']]
    df_precip = df_precip.pivot(index=['id', 'index'], columns='Date', values='precip').reset_index()
    df_precip = df_precip.rename_axis(None, axis=1)
    df_precip = df_precip.sort_values(['index', 'id'])
    # print(df_precip.shape)

    file_sp = '/home/kehuiyao/google_drive/SoilMoisture_Kehui/Climate/sp_grid_' + str(index) + '.csv'
    df_sp = pd.read_csv(file_sp)

    df_sp_columns = list(df_sp.columns)
    df_sp_columns[0] = 'Date'
    df_sp_columns[3] = 'sp'
    df_sp.columns = df_sp_columns
    df_sp['Date'] = pd.to_datetime(df_sp['Date'].apply(lambda x: x[:8])).astype(str)
    df_sp = df_sp[['id', 'index', 'Date', 'sp']]
    df_sp = df_sp.pivot(index=['id', 'index'], columns='Date', values='sp').reset_index()
    df_sp = df_sp.rename_axis(None, axis=1)
    df_sp = df_sp.sort_values(['index', 'id'])
    # print(df_sp.shape)

    file_tmin = '/home/kehuiyao/google_drive/SoilMoisture_Kehui/Climate/tmin_grid_' + str(index) + '.csv'
    df_tmin = pd.read_csv(file_tmin)

    df_tmin_columns = list(df_tmin.columns)
    df_tmin_columns[0] = 'Date'
    df_tmin_columns[3] = 'tmin'
    df_tmin.columns = df_tmin_columns
    df_tmin['Date'] = pd.to_datetime(df_tmin['Date'].apply(lambda x: x[:8])).astype(str)
    df_tmin = df_tmin[['id', 'index', 'Date', 'tmin']]
    df_tmin = df_tmin.pivot(index=['id', 'index'], columns='Date', values='tmin').reset_index()
    df_tmin = df_tmin.rename_axis(None, axis=1)
    df_tmin = df_tmin.sort_values(['index', 'id'])
    # print(df_tmin.shape)

    file_tmax = '/home/kehuiyao/google_drive/SoilMoisture_Kehui/Climate/tmax_grid_' + str(index) + '.csv'
    df_tmax = pd.read_csv(file_tmax)

    df_tmax_columns = list(df_tmax.columns)
    df_tmax_columns[0] = 'Date'
    df_tmax_columns[3] = 'tmax'
    df_tmax.columns = df_tmax_columns
    df_tmax['Date'] = pd.to_datetime(df_tmax['Date'].apply(lambda x: x[:8])).astype(str)
    df_tmax = df_tmax[['id', 'index', 'Date', 'tmax']]
    df_tmax = df_tmax.pivot(index=['id', 'index'], columns='Date', values='tmax').reset_index()
    df_tmax = df_tmax.rename_axis(None, axis=1)
    df_tmax = df_tmax.sort_values(['index', 'id'])
    # print(df_tmax.shape)

    file_wind = '/home/kehuiyao/google_drive/SoilMoisture_Kehui/Climate/wind_grid_' + str(index) + '.csv'
    df_wind = pd.read_csv(file_wind)

    df_wind_columns = list(df_wind.columns)
    df_wind_columns[0] = 'Date'
    df_wind_columns[3] = 'wind'
    df_wind.columns = df_wind_columns
    df_wind['Date'] = pd.to_datetime(df_wind['Date'].apply(lambda x: x[:8])).astype(str)
    df_wind = df_wind[['id', 'index', 'Date', 'wind']]
    df_wind = df_wind.pivot(index=['id', 'index'], columns='Date', values='wind').reset_index()
    df_wind = df_wind.rename_axis(None, axis=1)
    df_wind = df_wind.sort_values(['index', 'id'])

    smap_1km = df_smap_1km.loc[:, time_start: time_end].values  # K x L
    smap_36km = df_smap_36km.loc[:, time_start: time_end].values  # K x L

    dpt = df_dpt.loc[:, time_start: time_end].values  # K x L
    precip = df_precip.loc[:, time_start: time_end].values  # K x L
    sp = df_sp.loc[:, time_start: time_end].values   # K x L
    tmin = df_tmin.loc[:, time_start: time_end].values  # K x L
    tmax = df_tmax.loc[:, time_start: time_end].values  # K x L
    wind = df_wind.loc[:, time_start: time_end].values  # K x L

    static = df_static.loc[:, 'elevation':'soc'].values  # static features
    LC_one_hot_vec = one_hot_encoder.transform(df_static.loc[:, ['LC_36km', 'LC_1km']]).toarray()
    static = np.concatenate([static, LC_one_hot_vec], axis=1)

    y = smap_1km
    X_static = static
    X_varying = np.stack([smap_36km, \
                          dpt, \
                          precip, \
                          sp, \
                          tmin, \
                          tmax, \
                          wind], axis=2)

    return y, X_static, X_varying

def save_data():
    training_data_index = [  3,   5,  12, 268,  15,  20,  22, 288, 290, 291, 296, 297, 298,
        46, 304, 316, 320,  66, 323,  73,  84,  93, 353, 376, 122, 127,
       384, 138, 401, 150, 407, 408, 165, 166, 179, 438, 183, 185, 446,
       448, 197, 459, 460, 224, 485, 486, 231, 487, 232, 491, 493, 253,
       255,  23,  27,  32, 285, 294, 243, 247, 256, 432, 444, 457]

    testing_data_index = [269,  14, 287, 308, 352, 409, 451, 199, 481,  36, 362, 245]

    data_index = training_data_index + testing_data_index

    for index in data_index:
        print('process index %d' % index)
        y, X_static, X_varying = process_raw_sm_data(index)

        np.save('../data/y_' + str(index) + '.npy', y)
        np.save('../data/X_static_' + str(index) + '.npy', X_static)
        np.save('../data/X_varying_' + str(index) + '.npy', X_varying)


def visualize_index(ind, n_points=1):
    y = np.load('../soil_moisture/data/y_' + str(ind) + '.npy')  # 1296, 1095
    X_varying = np.load('../soil_moisture/data/X_varying_' + str(ind) + '.npy')  # 1296, 1095, 7
    X_static = np.load('../soil_moisture/data/X_static_' + str(ind) + '.npy')  # 1296, 35
    K, L = y.shape

    # randomly select some spatial points
    np.random.seed(0)
    idx = np.random.choice(y.shape[0], n_points, replace=False)
    y = y[idx, :]
    smap_36km = X_varying[idx, :, 0]
    dpt = X_varying[idx, :, 1]
    precip = X_varying[idx, :, 2]
    sp = X_varying[idx, :, 3]
    tmin = X_varying[idx, :, -3]
    tmax = X_varying[idx, :, -2]  # if tmin -273.15 < 4, then the ground soil moisture is not sensible
    wind = X_varying[idx, :, -1]

    plt.rcParams["font.size"] = 16

    for k in range(n_points):
        fig, axes = plt.subplots(nrows=8, ncols=1, sharex=True, constrained_layout=True, figsize=(24.0, 36.0))
        fig.delaxes(axes[-1])

        axes[0].scatter(np.arange(L), y[k, :], color='r')
        axes[1].scatter(np.arange(L), smap_36km[k, :], color='r')
        axes[2].scatter(np.arange(L), dpt[k, :], color='g')
        axes[3].scatter(np.arange(L), precip[k, :], color='b')
        axes[4].scatter(np.arange(L), sp[k, :], color='b')
        axes[5].scatter(np.arange(L), tmin[k, :], color='b')
        axes[5].axhline(y=4 + 273.15, color='k', linestyle='--')  # plot a horizontal line at 4 + 273.15
        axes[6].scatter(np.arange(L), tmax[k, :], color='b')
        axes[7].scatter(np.arange(L), wind[k, :], color='b')

        axes[0].set_ylabel('ground soil moisture')
        axes[1].set_ylabel('smap 36km')
        axes[2].set_ylabel('dpt')
        axes[3].set_ylabel('precip')
        axes[4].set_ylabel('sp')
        axes[5].set_ylabel('tmin')
        axes[6].set_ylabel('tmax')
        axes[7].set_ylabel('wind')

        # Display the plot
        # plt.show()
        # save figure
        fig.savefig('../soil_moisture/figures/visualize_index_' + str(ind) + '_' + str(k) + '.png', dpi=300)

        # close figure
        plt.close(fig)
        plt.clf()


def prepare_data(is_train=True):
    rng = np.random.RandomState(0)

    if is_train:
        data_index = [  3,   5,  12, 268,  15,  20,  22, 288, 290, 291, 296, 297, 298,
            46, 304, 316, 320,  66, 323,  73,  84,  93, 353, 376, 122, 127,
           384, 138, 401, 150, 407, 408, 165, 166, 179, 438, 183, 185, 446,
           448, 197, 459, 460, 224, 485, 486, 231, 487, 232, 491, 493, 253,
           255,  23,  27,  32, 285, 294, 243, 247, 256, 432, 444, 457]
    else:
        data_index = [269, 14, 287, 308, 352, 409, 451, 199, 481, 36, 362, 245]

    # randomly choose from training_data_index
    # data_index = np.random.choice(training_data_index, num_index, replace=False)

    # load data
    y_list = []
    X_list = []
    K = 1296
    L = 1095
    new_order = reorder_spatial_index(36, 36, 6, 6)
    new_order = new_order[:, 0] * 36 + new_order[:, 1]
    for ind in data_index:
        # print ind
        print('-----------------processing index: ', ind, '-----------------')
        y = np.load('../soil_moisture/data/y_' + str(ind) + '.npy')  # 1296, 1095
        # print y's shape
        print('data shape: ', y.shape)

        X_varying = np.load('../soil_moisture/data/X_varying_' + str(ind) + '.npy')  # 1296, 1095, 7
        X_static = np.load('../soil_moisture/data/X_static_' + str(ind) + '.npy')  # 1296, 35

        # if spatial dimension is not 36x36=1296, discard this data
        if y.shape[0] != K:
            print('spatial dimension is not 36x36=1296, discard this data')
            continue

        # reorder
        y = y[new_order, :L]  # K, L
        X_varying = X_varying[new_order, :L, :]  # K, L, *
        # repeat X_static to match the shape of X_varying
        X_static = np.repeat(X_static[new_order, np.newaxis, :], L, axis=1)  # K, L, *

        # y is three years soil moisture data, first of all, we split into three chunks
        y = np.split(y, 3, axis=1)
        X_varying = np.split(X_varying, 3, axis=1)
        X_static = np.split(X_static, 3, axis=1)

        seq_len = 128  # the length of each time series

        for i in range(3):
            print('process chunk %d' % i)
            # if tmin (one of the covariate in X_varying) is smaller than 4, make the corresponding y as nan
            non_frozen_mask = X_varying[i][0, :, -3] - 273.15 > 4  # L//3
            # for each chunk, we discard some data
            y[i] = y[i][:, non_frozen_mask]  # K, *
            X_varying[i] = X_varying[i][:, non_frozen_mask, :]  # K, *, *
            X_static[i] = X_static[i][:, non_frozen_mask, :]  # K, *, *

            # if the chunk size is larger than seq_len, randomly select a time series of length seq_len
            print('length of chunk %d is %d' % (i, y[i].shape[1]))
            if y[i].shape[1] > seq_len:
                start_ind = rng.randint(0, y[i].shape[1] - seq_len)
                y_list.append(y[i][:, start_ind:start_ind + seq_len])
                temp = np.concatenate([X_varying[i][:, start_ind:start_ind + seq_len, :],
                                       X_static[i][:, start_ind:start_ind + seq_len, :]], axis=-1)
                X_list.append(temp)

    y = np.stack(y_list, axis=0)  # B, K, seq_len
    X = np.stack(X_list, axis=0)  # B, K, seq_len, C
    B = y.shape[0]
    C = X.shape[-1]

    # reshape
    y = y.reshape(B, 36, 36, seq_len)  # B, 36, 36, seq_len
    y = y.reshape(B * 36, 36, seq_len)  # B*36, 36, seq_len, which is B*, K*, L*

    X = X.reshape(B, 36, 36, seq_len, C)  # B, 36, 36, seq_len, C
    X = X.reshape(B * 36, 36, seq_len, C)  # B*36, 36, seq_len, C, which is B*, K*, L*, C

    # standardize y_train
    y_mean = np.nanmean(y)
    y_std = np.nanstd(y)
    y_standardized = (y - y_mean) / y_std

    print('y shape is:', y_standardized.shape)

    observation_mask = ~np.isnan(y_standardized)  # True means observed, False means missing

    # standardize X_train for each feature C
    X_mean = np.nanmean(X, axis=(0, 1, 2))
    X_std = np.nanstd(X, axis=(0, 1, 2))
    X_standardized = (X - X_mean) / (X_std + 1e-8)

    # fill all missing values in X_standardized with 0
    X_standardized[np.isnan(X_standardized)] = 0

    print('X shape is: ', X_standardized.shape)

    # save y_standardized, X_standardized, y_mean, y_std, observation_mask in one file
    if is_train:
        np.savez('./data/dataset_train_gsm.npz', y_standardized=y_standardized, X_standardized=X_standardized,
                 y_mean=y_mean, y_std=y_std, observation_mask=observation_mask)
    else:
        np.savez('./data/dataset_test_gsm.npz', y_standardized=y_standardized, X_standardized=X_standardized,
                 y_mean=y_mean, y_std=y_std, observation_mask=observation_mask)

    return y_standardized, y_mean, y_std, X_standardized, observation_mask


def obtain_a_subset_of_data():
    data = np.load('./data/dataset_train_gsm.npz')
    y_standardized = data['y_standardized']  # B, K, L
    X_standardized = data['X_standardized']  # B, K, L, C
    observation_mask = data['observation_mask']  # B, K, L
    y_mean = data['y_mean']
    y_std = data['y_std']

    # randomly select 1/10 of the data
    rng = np.random.RandomState(0)
    B = y_standardized.shape[0]
    ind = rng.choice(B, B // 10, replace=False)
    y_standardized = y_standardized[ind]
    X_standardized = X_standardized[ind]
    observation_mask = observation_mask[ind]

    np.savez('./data/dataset_train_subset_gsm.npz', y_standardized=y_standardized, X_standardized=X_standardized,
             y_mean=y_mean, y_std=y_std, observation_mask=observation_mask)

def get_global_soil_moisture_data():
    data = np.load('./data/dataset_train_gsm.npz')
    y_standardized = data['y_standardized']  # B, K, L
    X_standardized = data['X_standardized']  # B, K, L, C
    observation_mask = data['observation_mask']  # B, K, L
    y_mean = data['y_mean']
    y_std = data['y_std']
    y_mean = float(y_mean)
    y_std = float(y_std)
    return y_standardized, y_mean, y_std, X_standardized, observation_mask

def get_global_soil_moisture_data_subset():
    data = np.load('./data/dataset_train_subset_gsm.npz')
    y_standardized = data['y_standardized']  # B, K, L
    X_standardized = data['X_standardized']  # B, K, L, C
    observation_mask = data['observation_mask']  # B, K, L
    y_mean = data['y_mean']
    y_std = data['y_std']
    y_mean = float(y_mean)
    y_std = float(y_std)
    return y_standardized, y_mean, y_std, X_standardized, observation_mask




if __name__ == '__main__':
    save_data()  # save y, X_varying, X_static for each index
    prepare_data(is_train=True)  # save y_standardized, X_standardized, y_mean, y_std, observation_mask in one file for all training indices
    prepare_data(is_train=False)  # save y_standardized, X_standardized, y_mean, y_std, observation_mask in one file for all test indices
    # load data
    data = np.load('./data/dataset_train_gsm.npz')
    y_standardized = data['y_standardized']  # B, K, L
    X_standardized = data['X_standardized']  # B, K, L, C
    observation_mask = data['observation_mask']  # B, K, L
    y_mean = data['y_mean']
    y_std = data['y_std']

    # select a random subset of the data
    obtain_a_subset_of_data()

