from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


class ImputationDataset(Dataset):
    def __init__(self, y, X, observation_mask, gt_mask, for_pattern_mask=None):

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

    # randomly mask some data as missing to test the performance of the model
    rng = np.random.RandomState(seed=seed)
    B, L, K = y.shape

    if missing_pattern == 'random':
        missing_mask = (rng.uniform(size=y.shape) > missing_data_ratio) & observation_mask  # missing mask equals 1 means observed, 0 means missing
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
        missing_mask = np.ones_like(y, dtype=bool)
        temp = np.argwhere(rng.uniform(size=y.shape[2]) < missing_data_ratio).flatten()

        for i in range(y.shape[0]):
            for j in range(y.shape[2]):
                if j in temp:
                    missing_mask[i, -10:, j] = False
        missing_mask = missing_mask & observation_mask


    missing_data_ratio = np.sum(~missing_mask) / missing_mask.size
    print(missing_data_ratio)  # print the actual missing data ratio, including the original missing plus the artificially missing
    # copy y to y_missing
    y_missing = y.copy()

    # replace all missing values and artificially missing values with 0
    y_missing[~missing_mask] = 0

    # create gt_mask, which select some data in the remaining non-missing data as validation data
    if missing_pattern == 'random':
        gt_mask = (rng.uniform(size=y_missing.shape) < training_data_ratio) & missing_mask  # gt_mask True means the data is used for training, False means the data is used for validation
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
        gt_mask = np.ones_like(y_missing, dtype=bool)
        temp = np.argwhere(rng.uniform(size=y.shape[2]) > training_data_ratio).flatten()
        for i in range(y.shape[0]):
            for j in range(y.shape[2]):
                if j in temp:
                    gt_mask[i, -10:, j] = False
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
