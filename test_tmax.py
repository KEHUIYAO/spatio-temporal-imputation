import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np



from dataset import get_dataloader
from dataset_tmax import generate_tmax_data
from birnn import BiRNN
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_tmax.yaml")
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
    "./save/tmax_birnn" + str(args.validationindex) + "_" + current_time + "/"
)

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
y, y_mean, y_std = generate_tmax_data()
train_loader, \
    valid_loader, \
    test_loader, \
    scaler, \
    mean_scaler = get_dataloader(y,
                              y_mean,
                              y_std,
                              missing_data_ratio=0.2,
                              training_data_ratio=0.8,
                              batch_size=config["train"]["batch_size"],
                              device=device,
                              missing_pattern=config['model']['missing_pattern'])


model = BiRNN(0, config, device).to(device)

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
