import argparse
import torch
import datetime
import json
import yaml
import os
from scipy.spatial.distance import pdist, squareform
import numpy as np

from dataset_synthetic import generate_ST_data_with_separable_covariance
from dataloader import get_dataloader
from main_model import CSDI_Covariates
from birnn import BiRNN
from simple_imputer import MeanImputer, LinearInterpolationImputer, KrigingImputer
from utils import train, evaluate

# missing_data_ratio_candidates = [0.1, 0.5, 0.9]
# missing_pattern_candidates = ['random', 'block', 'space_block', 'time_block']
# model_candidates = ['mean', 'interpolation', 'birnn', 'CSDI', 'Kriging']
# time_layer_candidates = [None, 'bilstm', 'transformer', 'longformer']
# spatial_layer_candidates = ['None', 'diffconv']

missing_data_ratio_candidates = [0.1, 0.5, 0.9]
missing_pattern_candidates = ['random', 'block']
model_candidates = ['Kriging']


K = 36
L = 36
B = 32

y, y_mean, y_std, adjacency_matrix, spatio_temporal_covariance_matrix = generate_ST_data_with_separable_covariance(K, L, B, seed=42)
# save adjacency matrix to data/adjaency_matrix.npy
np.save('data/adj_matrix.npy', adjacency_matrix)
training_data_ratio = 0.8
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nsample = 10

with open("config/base_synthetic.yaml", "r") as f:
    config = yaml.safe_load(f)


for missing_data_ratio in missing_data_ratio_candidates:
    for missing_pattern in missing_pattern_candidates:
        for model in model_candidates:
            print('------------------')
            print('missing_data_ratio:', missing_data_ratio)
            print('missing_pattern:', missing_pattern)
            print('model:', model)

            # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            foldername = (
                    "./save/synthetic_ST_separable" + '_' + str(missing_data_ratio) + "_" + str(missing_pattern) + "_" + str(model) + "/"
            )

            print('model folder:', foldername)
            os.makedirs(foldername, exist_ok=True)

            train_loader, \
                valid_loader, \
                test_loader, \
                scaler, \
                mean_scaler = get_dataloader(
                 y,
                 y_mean,
                 y_std,
                 missing_data_ratio=missing_data_ratio,
                 training_data_ratio=training_data_ratio,
                 batch_size=batch_size,
                 device=device,
                 missing_pattern=missing_pattern
                )

            config['model']['missing_pattern'] = missing_pattern


            if model == 'CSDI':
                model = CSDI_Covariates(config, device, target_dim=K, covariate_dim=0).to(device)
                # you can change config here to train different models
                train(
                    model,
                    config["train"],
                    train_loader,
                    valid_loader=valid_loader,
                    foldername=foldername,
                )

            elif model == 'birnn':
                model = BiRNN(covariate_size=0, config=config, device=device).to(device)
                train(
                    model,
                    config["train"],
                    train_loader,
                    valid_loader=valid_loader,
                    foldername=foldername,
                )

            elif model == 'mean':
                model = MeanImputer(device=device)

            elif model == 'interpolation':
                model = LinearInterpolationImputer(device=device)

            elif model == 'Kriging':
                model = KrigingImputer(device=device, mu=np.zeros(K*L), covariance_matrix=spatio_temporal_covariance_matrix)


            # evaluate the model
            evaluate(
                model,
                test_loader,
                nsample=nsample,
                scaler=scaler,
                mean_scaler=mean_scaler,
                foldername=foldername
            )
