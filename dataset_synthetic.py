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

    rng = np.random.RandomState(seed)
    for b in range(B):
        # Spatial and temporal autoregressive coefficients
        spatial_ar_coeff = 0.6
        temporal_ar_coeff = 0.7

        # Compute spatial correlation matrix
        spatial_distances = squareform(pdist(spatial_coords))
        spatial_correlation = np.exp(-spatial_distances / 0.1)

        # normalize spatial correlation matrix for each row
        spatial_correlation = spatial_correlation / spatial_correlation.sum(axis=1, keepdims=True)

        # Generate synthetic temperature data
        temp_data = np.ones((len(spatial_coords), n_days)) * 20

        # Add randomness to the observations
        independent_noise = rng.normal(0, 1, temp_data.shape)
        spatial_correlated_noise = rng.normal(0, 1, temp_data.shape)
        correlated_noise_matrix = np.dot(spatial_correlation, spatial_correlated_noise)

        # Generate autoregressive (AR) error term
        ar_coeff = 1
        ar_error = np.zeros(temp_data.shape)
        for t in range(1, n_days):
            # ar_error[:, t] = ar_coeff * ar_error[:, t - 1] + 1 + 0.2 * rng.normal(0, 1, len(spatial_coords))
            ar_error[:, t] = ar_coeff * ar_error[:, t - 1] + 1 + 0.5 * rng.normal(0, 1, len(spatial_coords))


        # temp_data_with_noise = temp_data + independent_noise + correlated_noise_matrix + ar_error
        temp_data_with_noise = temp_data + ar_error

        # Store the temperature and feature data
        temperature_data[b] = temp_data_with_noise

    y = temperature_data  # shape (B, K, L)

    # standardize y using np.nanstd and np.nanmean
    y_std = np.nanstd(y)
    y_mean = np.nanmean(y)
    y_standardized = (y - y_mean) / y_std

    observation_mask = ~np.isnan(y_standardized)  # True means observed, False means missing

    return y_standardized, y_mean, y_std, observation_mask

