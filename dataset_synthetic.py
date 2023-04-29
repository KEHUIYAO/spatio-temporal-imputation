import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

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

    return temperature_data, feature_data


class SyntheticDataset(Dataset):
    def __init__(self, y, X, observation_mask, gt_mask, for_pattern_mask=None):

        # numpy, permute to B x K x L
        self.y = np.transpose(y, (0, 2, 1))
        self.observation_mask = np.transpose(observation_mask, (0, 2, 1))
        self.gt_mask = np.transpose(gt_mask, (0, 2, 1))

        # permute (B, L, K, C) to (B, C, K, L)
        self.X = np.transpose(X, (0, 3, 2, 1))

        if for_pattern_mask is not None:
            self.for_pattern_mask = np.transpose(for_pattern_mask, (0, 2, 1))
        else:
            self.for_pattern_mask = None


    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        y = torch.tensor(self.y[idx, ...], dtype=torch.float32)
        observation_mask = torch.tensor(self.observation_mask[idx, ...], dtype=torch.bool)
        batch = {}
        batch['observed_data'] = y
        batch['observed_mask'] = observation_mask
        batch['timepoints'] = torch.arange(y.shape[1], dtype=torch.float32)
        batch['gt_mask'] = torch.tensor(self.gt_mask[idx, ...], dtype=torch.bool)
        batch['observed_covariates'] = torch.tensor(self.X[idx, ...], dtype=torch.float32)

        if self.for_pattern_mask is not None:
            batch['for_pattern_mask'] = torch.tensor(self.for_pattern_mask[idx, ...], dtype=torch.bool)
        return batch


def get_dataloader(missing_data_ratio, training_data_ratio, batch_size, device, seed=42, missing_pattern='random'):
    """
    generate synthetic spatial-temporal data with random missingness.
    :param missing_data_ratio:
    :param training_data_ratio:
    :param batch_size:
    :param device:
    :param seed:
    :param missing_pattern:
    :return:
    """

    B = 40
    n_lat, n_lon = 6, 6
    n_days = 36

    # Generate synthetic data
    temperature_data, feature_data = generate_synthetic_data(n_lat, n_lon, n_days, B)
    # reshape temperature_data to have shape (B, L, K)
    y = np.transpose(temperature_data, (0, 2, 1))
    # reshape feature_data to have shape (B, L, K, C)
    X = np.transpose(feature_data, (0, 2, 1, 3))

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
    B, L, K = y_standardized.shape

    if missing_pattern == 'random':
        missing_mask = (rng.uniform(size=y_standardized.shape) > missing_data_ratio) & observation_mask  # randomly mask some data as missing
    elif missing_pattern == 'block':
        # Define the range of block sizes for missing values in the temporal axis
        min_block_size = 5
        max_block_size = 20

        # Create a mask with the same shape as the original array
        mask = np.ones((B, L, K), dtype=bool)

        # Randomly select the starting position and block size for the block along the last axis
        block_sizes = rng.randint(min_block_size, max_block_size + 1, (B, K))
        start_positions = [rng.randint(0, L - block_size + 1) for block_size in block_sizes.ravel()]
        start_positions = np.array(start_positions).reshape(B, K)

        # Set the desired block of missing values in the mask along the last axis
        for b in range(B):
            for k in range(K):
                mask[b, start_positions[b, k]:start_positions[b, k] + block_sizes[b, k], k] = False
        temporal_mask = mask.copy()

        # Define the range of block sizes for missing values in the spatial axis
        min_block_size = 5
        max_block_size = 20

        # Create a mask with the same shape as the original array
        mask = np.ones((B, L, K), dtype=bool)

        # Randomly select the starting position and block size for the block along the last axis
        block_sizes = rng.randint(min_block_size, max_block_size + 1, (B, L))
        start_positions = [rng.randint(0, K - block_size + 1) for block_size in block_sizes.ravel()]
        start_positions = np.array(start_positions).reshape(B, L)

        # Set the desired block of missing values in the mask along the last axis
        for b in range(B):
            for l in range(L):
                mask[b, l, start_positions[b, l]:start_positions[b, l] + block_sizes[b, l]] = False
        spatial_mask = mask.copy()

        missing_mask = temporal_mask & spatial_mask & observation_mask

    elif missing_pattern == 'space_block':  # at a given time point, all spatial locations are missing
        temp = rng.uniform(
            size=(B, L, 1)) > missing_data_ratio  # all spatial points are either observed or missing at tone time point
        missing_mask = np.repeat(temp, K, axis=2) & observation_mask  # randomly mask some data as missing


    elif missing_pattern == 'time_block':  # at a given spatial location, all time points are missing
        temp = rng.uniform(size=(B, 1, K)) > missing_data_ratio  # all time points are either observed or missing at one spatial location
        missing_mask = np.repeat(temp, L, axis=1) & observation_mask  # randomly mask some data as missing

    elif missing_pattern == 'prediction':
        # randomly mask the last 20% time steps of the data as missing
        missing_mask = np.ones_like(y_standardized, dtype=bool)
        temp = np.argwhere(rng.uniform(size=y_standardized.shape[2]) < missing_data_ratio).flatten()

        for i in range(y_standardized.shape[0]):
            for j in range(y_standardized.shape[2]):
                if j in temp:
                    missing_mask[i, -10:, j] = False
        missing_mask = missing_mask & observation_mask


    missing_data_ratio = np.sum(~missing_mask) / missing_mask.size
    print(missing_data_ratio)
    # copy y_standardized to y_missing_standardized
    y_missing_standardized = y_standardized.copy()

    # replace all missing values and artificially missing values with 0
    y_missing_standardized[~missing_mask] = 0

    # create gt_mask, which additionally mask some data in the remaining non-missing data
    if missing_pattern == 'random':
        gt_mask = (rng.uniform(size=y_missing_standardized.shape) < training_data_ratio) & missing_mask  # gt_mask True means used for training
    elif missing_pattern == 'block':
        pass
    elif missing_pattern == 'space_block':
        temp = rng.uniform(
            size=(B, L, 1)) < training_data_ratio  # all spatial points are either observed or missing at tone time point
        gt_mask = np.repeat(temp, K, axis=2) & missing_mask  # randomly mask some data as missing

    elif missing_pattern == 'time_block':
        temp = rng.uniform(size=(
        B, 1, K)) < training_data_ratio  # all time points are either observed or missing at one spatial location
        gt_mask = np.repeat(temp, L, axis=1) & missing_mask  # randomly mask some data as missing

    elif missing_pattern == 'prediction':
        gt_mask = np.ones_like(y_missing_standardized, dtype=bool)
        temp = np.argwhere(rng.uniform(size=y_standardized.shape[2]) > training_data_ratio).flatten()
        for i in range(y_standardized.shape[0]):
            for j in range(y_standardized.shape[2]):
                if j in temp:
                    gt_mask[i, -10:, j] = False
        gt_mask = gt_mask & missing_mask

    # gt_mask True for training data
    y_training = y_missing_standardized.copy()
    y_training[~gt_mask] = 0

    # create pattern mask
    if missing_pattern == 'random':
        pattern_mask = rng.uniform(size=y_missing_standardized.shape) > missing_data_ratio
    elif missing_pattern == 'block':
        pattern_mask = None
    elif missing_pattern == 'space_block':
        temp = rng.uniform(
            size=(
            B, L, 1)) < training_data_ratio  # all spatial points are either observed or missing at tone time point
        pattern_mask = np.repeat(temp, K, axis=2)

    elif missing_pattern == 'time_block':
        temp = rng.uniform(size=(
            B, 1, K)) < training_data_ratio  # all time points are either observed or missing at one spatial location
        pattern_mask = np.repeat(temp, L, axis=1)

    elif missing_pattern == 'prediction':
        pattern_mask = np.ones_like(y_missing_standardized, dtype=bool)
        pattern_mask[:, -int(0.2 * n_days):, :] = False

    training_dataset = SyntheticDataset(y_training, X_standardized, gt_mask, gt_mask, pattern_mask)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = SyntheticDataset(y_missing_standardized, X_standardized, missing_mask, gt_mask)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # replace missing values in the real data with 0
    y_standardized[~observation_mask] = 0

    # create test dataset
    # we use the complete data as the test data
    test_dataset = SyntheticDataset(y_standardized, X_standardized, observation_mask, missing_mask)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return training_loader, validation_loader, test_loader, y_std, y_mean





if __name__ == "__main__":
    missing_data_ratio = 0.5
    training_data_ratio = 0.8
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    get_dataloader(missing_data_ratio, training_data_ratio, batch_size, device, seed=42, missing_pattern='prediction')
