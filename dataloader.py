from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def generate_synthetic_data(n_lat, n_lon, n_days, B, seed=42):
    latitudes = np.linspace(0, 1, n_lat)
    longitudes = np.linspace(0, 1, n_lon)
    spatial_coords = np.array(np.meshgrid(latitudes, longitudes)).T.reshape(-1, 2)
    timestamps = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    temperature_data = np.zeros((B, len(spatial_coords), n_days))
    feature_data = np.zeros((B, len(spatial_coords), n_days, 3))

    rng = np.random.RandomState(seed)
    for b in range(B):
        # Generate additional covariates
        elevation = rng.normal(0, 1, len(spatial_coords))
        distance_to_water = rng.uniform(0, 1, len(spatial_coords))

        # Generate wind speed data with spatial and temporal dependencies
        wind_speed = np.zeros((len(spatial_coords), n_days))
        wind_speed[:, 0] = rng.normal(0, 1, len(spatial_coords))

        # Spatial and temporal autoregressive coefficients
        spatial_ar_coeff = 0.6
        temporal_ar_coeff = 0.7

        # Compute spatial correlation matrix
        spatial_distances = squareform(pdist(spatial_coords))
        spatial_correlation = np.exp(-spatial_distances / 0.1)

        # normalize spatial correlation matrix for each row
        spatial_correlation = spatial_correlation / spatial_correlation.sum(axis=1, keepdims=True)


        # Generate spatial and temporal autoregressive wind speed data
        for t in range(1, n_days):
            wind_speed[:, t] = temporal_ar_coeff * wind_speed[:, t - 1] + rng.normal(0, 1, len(spatial_coords))


        # Define temperature function with wind speed as a factor
        def temperature_function(lat, lon, day, elevation, distance_to_water, wind_speed):
            return 20 + elevation + distance_to_water + wind_speed

        # Generate synthetic temperature data
        temp_data = np.zeros((len(spatial_coords), n_days))

        # location specific random effect
        location_specific_random_effect = rng.normal(0, 1, len(spatial_coords))


        for i, (lat, lon) in enumerate(spatial_coords):
            for j, timestamp in enumerate(timestamps):
                temp_data[i, j] = temperature_function(lat, lon, timestamp.dayofyear, elevation[i], distance_to_water[i], wind_speed[i, j]) + location_specific_random_effect[i]

        # Add randomness to the observations
        independent_noise = rng.normal(0, 1, temp_data.shape)
        spatial_correlated_noise = rng.normal(0, 5, temp_data.shape)
        correlated_noise_matrix = np.dot(spatial_correlation, spatial_correlated_noise)

        # Generate autoregressive (AR) error term
        ar_coeff = 0.7
        ar_error = np.zeros(temp_data.shape)
        for t in range(1, n_days):
            ar_error[:, t] = ar_coeff * ar_error[:, t - 1] + rng.normal(0, 1, len(spatial_coords))

        temp_data_with_noise = temp_data + independent_noise + correlated_noise_matrix + ar_error

        # Store the temperature and feature data
        temperature_data[b] = temp_data_with_noise
        feature_data[b, :, :, 0] = elevation.reshape(-1, 1)
        feature_data[b, :, :, 1] = distance_to_water.reshape(-1, 1)
        feature_data[b, :, :, 2] = wind_speed

        y = temperature_data  # shape (B, K, L)
        X = feature_data  # shape (B, K, L, C)

        # standardize y using np.nanstd and np.nanmean
        y_std = np.nanstd(y)
        y_mean = np.nanmean(y)
        y_standardized = (y - y_mean) / y_std

        # standardize X using np.nanstd and np.nanmean for each feature c
        X_std = np.nanstd(X, axis=(0, 1, 2))  # shape (C,)
        X_mean = np.nanmean(X, axis=(0, 1, 2))  # shape (C,)
        X_standardized = (X - X_mean) / X_std

        observation_mask = ~np.isnan(y_standardized)  # True means observed, False means missing

    return y_standardized, y_mean, y_std, X_standardized, observation_mask



def get_soilmoisture_TxSon():
    # read data from data/SMAP_Climate_In_Situ_TxSon.csv
    df = pd.read_csv("./data/Insitu_gap_filling_data.csv")
    # df = pd.read_csv("./data/SMAP_Climate_In_Situ_TxSon.csv")
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
    y = np.stack(temp, axis=0)  # shape (B*36, 108, 36)

    temp = np.vsplit(X[:B * T, :, :], B)
    temp = [np.hsplit(i, 36) for i in temp]  # split the sptial points into 36 for one field
    temp = [item for sublist in temp for item in sublist]
    temp = np.stack(temp, axis=0)  # each sample is a 6x6 grid with 108 time steps
    X = np.stack(temp, axis=0)  # shape (B*36, 108, 36, C)

    # reset B, L, K
    B = y.shape[0]
    L = y.shape[1]
    K = y.shape[2]

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
    X_standardized = np.concatenate([X_standardized, X_standardized_mask], axis=-1)  # shape (B, L, K, C)

    observation_mask = ~np.isnan(y_standardized)  # True means observed, False means missing

    # reshape to B, K, L
    y_standardized = np.transpose(y_standardized, (0, 2, 1))
    X_standardized = np.transpose(X_standardized, (0, 2, 1, 3))
    observation_mask = np.transpose(observation_mask, (0, 2, 1))

    return y_standardized, y_mean, y_std, X_standardized, observation_mask

class ImputationDataset(Dataset):
    def __init__(self, y, X, observation_mask, gt_mask, for_pattern_mask=None):
        """

        :param y: B, L, K
        :param X: B, L, K, C or None
        :param observation_mask: B, L, K
        :param gt_mask: B, L, K
        :param for_pattern_mask: B, L, K or None
        """

        # permute to B x K x L
        self.y = np.transpose(y, (0, 2, 1))
        self.observation_mask = np.transpose(observation_mask, (0, 2, 1))
        self.gt_mask = np.transpose(gt_mask, (0, 2, 1))

        if X is not None:
            # permute (B, L, K, C) to (B, C, K, L)
            self.X = np.transpose(X, (0, 3, 2, 1))
        else:
            self.X = None

        if for_pattern_mask is not None:
            # permute to B x K x L
            self.for_pattern_mask = np.transpose(for_pattern_mask, (0, 2, 1))
        else:
            self.for_pattern_mask = None


    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        batch = {}
        batch['observed_data'] = torch.tensor(self.y[idx, ...], dtype=torch.float32)
        batch['observed_mask'] = torch.tensor(self.observation_mask[idx, ...], dtype=torch.bool)
        batch['timepoints'] = torch.arange(self.y.shape[2], dtype=torch.float32)
        batch['gt_mask'] = torch.tensor(self.gt_mask[idx, ...], dtype=torch.bool)
        if self.for_pattern_mask is not None:
            batch['for_pattern_mask'] = torch.tensor(self.for_pattern_mask[idx, ...], dtype=torch.bool)

        if self.X is not None:
            batch['observed_covariates'] = torch.tensor(self.X[idx, ...], dtype=torch.float32)

        return batch


def get_dataloader(y,
                   y_mean,
                   y_std,
                   X=None,
                   observation_mask=None,
                   missing_mask=None,
                   missing_data_ratio=0.2,
                   training_data_ratio=0.8,
                   batch_size=16,
                   device='cpu',
                   seed=42,
                   missing_pattern='random',
                   *args, **kwargs):
    """
    generate synthetic spatial-temporal data with random missingness.
    :param y: (B, K, L), standardized y
    :param y_mean: mean of the original y
    :param y_std: std of the original y
    :param X: None or (B, K, L, C), standardized features
    :param observation_mask: None or (B, K, L), True means observed, False means missing
    :param missing_mask: None or (B, K, L), True means used for training, False means used for testing
    :param missing_data_ratio:
    :param training_data_ratio:
    :param batch_size:
    :param device:
    :param seed:
    :param missing_pattern:
    :return:
    """


    # reshape temperature_data to have shape (B, L, K)
    y = np.transpose(y, (0, 2, 1))

    # if feature data is provided
    if X is not None:
        # reshape feature_data to have shape (B, L, K, C)
        X = np.transpose(X, (0, 2, 1, 3))

    # if the observation mask is not provided, compute it
    if observation_mask is None:
        observation_mask = ~np.isnan(y)  # True means observed, False means missing
    else:
        observation_mask = np.transpose(observation_mask, (0, 2, 1))  # permute to B x L x K

    # randomly mask some data as missing to test the performance of the model
    rng = np.random.RandomState(seed=seed)
    B, L, K = y.shape

    if missing_mask is None:  # if there is no testing data provided, aritifically generate some
        print('generating missing data...')
        if missing_pattern == 'random':
            missing_mask = (rng.uniform(size=y.shape) > missing_data_ratio) & observation_mask  # missing mask equals 1 means observed, 0 means missing
        elif missing_pattern == 'block':
            missing_mask = np.ones_like(y, dtype=bool)
            expected_missing_observations = int(missing_data_ratio * B * L * K)  # expected number of missing observations
            # current number of missing observations
            cur_missing_obs = 0
            while cur_missing_obs < expected_missing_observations:
                # randomly select a block of data to be missing
                b = rng.randint(0, B)
                # if block size is provided in **kwargs, use it, otherwise randomly select a block size
                if 'time_block_size' in kwargs:
                    l_block_size = kwargs['time_block_size']
                else:
                    l_block_size = rng.randint(0, L + 1)
                if 'space_block_size' in kwargs:
                    k_block_size = kwargs['space_block_size']
                else:
                    k_block_size = rng.randint(0, K + 1)
                l_start = rng.randint(0, L - l_block_size + 1)
                k_start = rng.randint(0, K - k_block_size + 1)
                missing_mask[b, l_start:l_start + l_block_size, k_start:k_start + k_block_size] = False
                cur_missing_obs += l_block_size * k_block_size
            missing_mask = missing_mask & observation_mask

        elif missing_pattern == 'space_block':  # at a given time point, all spatial locations are missing
            temp = rng.uniform(
                size=(B, L, 1)) > missing_data_ratio  # all spatial points are either observed or missing at tone time point
            missing_mask = np.repeat(temp, K, axis=2) & observation_mask  # randomly mask some data as missing

        elif missing_pattern == 'time_block':  # at a given spatial location, all time points are missing
            temp = rng.uniform(size=(B, 1, K)) > missing_data_ratio  # all time points are either observed or missing at one spatial location
            missing_mask = np.repeat(temp, L, axis=1) & observation_mask  # randomly mask some data as missing

        elif missing_pattern == 'prediction':
            # randomly mask the last missing_ratio percent of time steps of the data as missing
            missing_mask = np.ones_like(y, dtype=bool)
            expected_missing_observations = int(missing_data_ratio * B * L * K)  # expected number of missing observations
            # current number of missing observations
            cur_missing_obs = 0

            while cur_missing_obs < expected_missing_observations:
                # randomly select a block of data to be missing
                b = rng.randint(0, B)
                l = rng.randint(0, L)
                k = rng.randint(0, K)
                missing_mask[b, l:L, k] = False
                cur_missing_obs += (L-l)

            missing_mask = missing_mask & observation_mask
    else:
        missing_mask = missing_mask.reshape(B, L, K)  # reshape missing_mask to have shape (B, L, K)
        print('using provided missing mask')
    print("actual missing ratio in data %.4f"%(np.sum(~observation_mask) / observation_mask.size))
    print("after artifically setting some data to test data, the missing ratio is %.4f"%(np.sum(~missing_mask) / missing_mask.size))

    # copy y to y_missing
    y_missing = y.copy()

    # replace all missing values and artificially missing values with 0
    y_missing[~missing_mask] = 0

    # create gt_mask, which select some data in the remaining non-missing data as validation data
    if missing_pattern == 'random':
        gt_mask = (rng.uniform(size=y_missing.shape) < training_data_ratio) & missing_mask  # gt_mask True means the data is used for training, False means the data is used for validation
    elif missing_pattern == 'block':
        gt_mask = np.ones_like(y, dtype=bool)
        expected_missing_observations = int((1-training_data_ratio) * B * L * K)  # expected number of missing observations
        # current number of missing observations
        cur_missing_obs = 0
        while cur_missing_obs < expected_missing_observations:
            # randomly select a block of data to be missing
            b = rng.randint(0, B)
            l_block_size = rng.randint(0, L + 1)
            k_block_size = rng.randint(0, K + 1)
            l_start = rng.randint(0, L - l_block_size + 1)
            k_start = rng.randint(0, K - k_block_size + 1)
            gt_mask[b, l_start:l_start + l_block_size, k_start:k_start + k_block_size] = False
            cur_missing_obs += l_block_size * k_block_size
        gt_mask = gt_mask & missing_mask


    elif missing_pattern == 'space_block':
        temp = rng.uniform(
            size=(B, L, 1)) < training_data_ratio  # all spatial points are either observed or missing at tone time point
        gt_mask = np.repeat(temp, K, axis=2) & missing_mask  # randomly mask some data as missing

    elif missing_pattern == 'time_block':
        temp = rng.uniform(size=(
        B, 1, K)) < training_data_ratio  # all time points are either observed or missing at one spatial location
        gt_mask = np.repeat(temp, L, axis=1) & missing_mask  # randomly mask some data as missing

    elif missing_pattern == 'prediction':
        # randomly mask the last missing_ratio percent of time steps of the data as missing
        gt_mask = np.ones_like(y, dtype=bool)
        expected_missing_observations = int((1-training_data_ratio) * B * L * K)  # expected number of missing observations
        # current number of missing observations
        cur_missing_obs = 0

        while cur_missing_obs < expected_missing_observations:
            # randomly select a block of data to be missing
            b = rng.randint(0, B)
            l = rng.randint(0, L)
            k = rng.randint(0, K)
            gt_mask[b, l:L, k] = False
            cur_missing_obs += (L - l)

        gt_mask = gt_mask & missing_mask


    # gt_mask True for training data
    y_training = y_missing.copy()
    y_training[~gt_mask] = 0

    # create pattern mask
    pattern_mask = missing_mask

    training_dataset = ImputationDataset(y_training, X, gt_mask, gt_mask, pattern_mask)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = ImputationDataset(y_missing, X, missing_mask, gt_mask)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # replace missing values in the real data with 0
    y[~observation_mask] = 0

    # create test dataset
    # we use the complete data as the test data
    test_dataset = ImputationDataset(y, X, observation_mask, missing_mask)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return training_loader, validation_loader, test_loader, y_std, y_mean
