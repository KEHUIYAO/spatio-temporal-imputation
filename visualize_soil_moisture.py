import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import io
import argparse
import os
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser(description="visualize soil moisture")
parser.add_argument("--folder", type=str, default='gsm_CSDI_20230502_230939')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def get_quantile(samples,q,dim=1):
    return torch.quantile(samples,q,dim=dim).cpu().numpy()

dataset = 'soilmoisture'
args = parser.parse_args()
datafolder = args.folder

nsample = 10  # number of generated samples
path = './save/' + datafolder + '/generated_outputs_nsample' + str(nsample) + '.pk'

if __name__ == '__main__':
    with open(path, 'rb') as f:
        if torch.cuda.is_available():
            samples, all_target, all_evalpoint, all_observed, all_observed_time, scaler, mean_scaler = pickle.load(f)
        else:
            samples, all_target, all_evalpoint, all_observed, all_observed_time, scaler, mean_scaler = CPU_Unpickler(
                f).load()

    all_target_np = all_target.cpu().numpy()
    all_evalpoint_np = all_evalpoint.cpu().numpy()
    all_observed_np = all_observed.cpu().numpy()
    all_given_np = all_observed_np - all_evalpoint_np

    B = samples.shape[0]  # batch size
    K = samples.shape[-1]  # feature
    L = samples.shape[-2]  # time length

    all_target_np = (all_target_np * scaler + mean_scaler)
    samples = (samples * scaler + mean_scaler)

    qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantiles_imp = []
    for q in qlist:
        quantiles_imp.append(get_quantile(samples, q, dim=1) * (1 - all_given_np) + all_target_np * all_given_np)

    # create a folder called 'figures' to save the figures
    foldername = './save/' + datafolder + '/figures/'
    os.makedirs(foldername, exist_ok=True)

    # visualize all the samples
    print('total number of samples: ', B)

    # create an progress bar
    for dataind in tqdm(range(B)):
        plt.rcParams["font.size"] = 16
        fig, axes = plt.subplots(nrows=9, ncols=4, figsize=(24.0, 36.0))
        fig.delaxes(axes[-1][-1])
        offset = 0

        for k in range(36):
            df = pd.DataFrame({"x": np.arange(0, L), "val": all_target_np[dataind, :, k + offset],
                               "y": all_evalpoint_np[dataind, :, k + offset]})
            df = df[df.y != 0]
            df2 = pd.DataFrame({"x": np.arange(0, L), "val": all_target_np[dataind, :, k + offset],
                                "y": all_given_np[dataind, :, k + offset]})
            df2 = df2[df2.y != 0]
            row = k // 4
            col = k % 4
            axes[row][col].plot(range(0, L), quantiles_imp[2][dataind, :, k + offset], color='g', linestyle='solid',
                                label='CSDI')
            axes[row][col].fill_between(range(0, L), quantiles_imp[0][dataind, :, k + offset],
                                        quantiles_imp[4][dataind, :, k + offset],
                                        color='g', alpha=0.3)
            axes[row][col].plot(df.x, df.val, color='b', marker='o', linestyle='None')
            axes[row][col].plot(df2.x, df2.val, color='r', marker='x', linestyle='None')
            if col == 0:
                plt.setp(axes[row, 0], ylabel='value')
            if row == -1:
                plt.setp(axes[-1, col], xlabel='time')

        # save the figure
        figname = foldername + 'sample_' + str(dataind) + '.png'
        plt.savefig(figname, bbox_inches='tight')
        plt.close(fig)
        plt.clf()

