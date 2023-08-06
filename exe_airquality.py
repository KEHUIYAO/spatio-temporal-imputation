import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np

from dataset_airquality import get_dataloader
from main_model import CSDI_Covariates
from birnn import BiRNN
from bigcrnn import BiGCRNN
from brits import BRITS
from grin import GRIN
from simple_imputer import MeanImputer, LinearInterpolationImputer
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--model", type=str, default='grin')
parser.add_argument("--config", type=str, default="base_airquality.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"]
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

foldername = (
    "./save/airquality" + "_" + str(args.model) + "/"
)

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    config["train"]["batch_size"], device=device
)

print(scaler)
# print(mean_scaler)

if args.model == 'CSDI':
    model = CSDI_Covariates(config, device, target_dim=36, covariate_dim=0).to(device)

elif args.model == 'birnn':
    model = BiRNN(covariate_size=0, config=config, device=device).to(device)

elif args.model == 'bigcrnn':
    adj = np.load('data/adj_matrix.npy')
    model = BiGCRNN(target_size=36, covariate_size=0, config=config, adj=adj, device=device).to(device)

elif args.model == 'brits':
    model = BRITS(n_steps=36, n_features=36, rnn_hidden_size=108, device=device, config=config).to(device)

elif args.model == 'mean':
    model = MeanImputer(device=device)

elif args.model == 'grin':
    adj = np.load('data/adj_matrix.npy')
    model = GRIN(adj=adj - np.eye(36),
                 d_in=1,
                 d_hidden=64,
                 d_ff=64,
                 ff_dropout=0,
                 config=config,
                 device=device).to(device)

elif args.model == 'interpolation':
    model = LinearInterpolationImputer(device=device)

# these models needs to be trained
if args.model in ['CSDI', 'birnn', 'bigcrnn', 'brits', 'grin']:
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