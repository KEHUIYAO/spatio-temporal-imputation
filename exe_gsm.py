import argparse
import torch
import datetime
import json
import yaml
import os
from scipy.spatial.distance import pdist, squareform
import numpy as np

from dataset_gsm import get_global_soil_moisture_data, get_global_soil_moisture_data_subset
from dataloader import get_dataloader
from main_model import CSDI_Covariates
from birnn import BiRNN
from simple_imputer import MeanImputer, LinearInterpolationImputer
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--model", type=str, default='mean')
parser.add_argument("--config", type=str, default="base_gsm.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--targetstrategy", type=str, default="random", choices=["mix", "random", "historical"]
)
parser.add_argument(
    "--validationindex", type=int, default=0, help="index of month used for validation (value:[0-7])"
)
parser.add_argument("--nsample", type=int, default=10)
parser.add_argument("--unconditional", action="store_true")

parser.add_argument("--missing_data_ratio", type=float, default=0.2)
parser.add_argument("--training_data_ratio", type=float, default=0.8)

parser.add_argument("--epochs", type=int, default=0)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

if args.epochs > 0:
    config["train"]["epochs"] = args.epochs

# config["model"]["is_unconditional"] = args.unconditional
# config["model"]["target_strategy"] = args.targetstrategy

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = (
    "./save/gsm" + '_' + str(args.model) + "_" + current_time + "/"
)

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


# generate adj matrix
latitudes = np.linspace(0, 1, 6)
longitudes = np.linspace(0, 1, 6)
spatial_coords = np.array(np.meshgrid(latitudes, longitudes)).T.reshape(-1, 2)

# adjacency matrix
spatial_distances = squareform(pdist(spatial_coords))
spatial_correlation = np.exp(-spatial_distances / 0.1)
adjacency_matrix = spatial_correlation

# save adjacency matrix to data/adjaency_matrix.npy
np.save('data/adj_matrix_soilmoisture.npy', adjacency_matrix)
# y, y_mean, y_std, X, observation_mask = get_global_soil_moisture_data_subset()
y, y_mean, y_std, X, observation_mask = get_global_soil_moisture_data()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, \
    valid_loader, \
    test_loader, \
    scaler, \
    mean_scaler = get_dataloader(y,
                              y_mean,
                              y_std,
                              X=X,
                              observation_mask=observation_mask,
                              missing_data_ratio=args.missing_data_ratio,
                              training_data_ratio=args.training_data_ratio,
                              batch_size=config["train"]["batch_size"],
                              device=device,
                              missing_pattern=config['model']['missing_pattern']
                                 )




if args.model == 'CSDI':
    model = CSDI_Covariates(config, device, target_dim=36, covariate_dim=42).to(device)

elif args.model == 'birnn':
    model = BiRNN(covariate_size=42, config=config, device=device).to(device)

elif args.model == 'mean':
    model = MeanImputer(device=device)

elif args.model == 'interpolation':
    model = LinearInterpolationImputer(device=device)

# these models needs to be trained
if args.model in ['CSDI', 'birnn']:
    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
# evaluate the model
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername
)