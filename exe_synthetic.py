import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform


from dataset_synthetic import get_dataloader
from main_model import CSDI_Covariates
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_synthetic.yaml")
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

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

# config["model"]["is_unconditional"] = args.unconditional
# config["model"]["target_strategy"] = args.targetstrategy

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = (
    "./save/synthetic_validationindex" + str(args.validationindex) + "_" + current_time + "/"
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
np.save('data/adj_matrix.npy', adjacency_matrix)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    missing_data_ratio=0.2,
    training_data_ratio=0.8,
    batch_size=config["train"]["batch_size"],
    device=args.device,
    missing_pattern=config['model']['missing_pattern']
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CSDI_Covariates(config, device, target_dim=36, covariate_dim=6).to(device)

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

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername
)
