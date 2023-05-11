# Main Result

Model: $Y(s,t) = \eta(s, t)+\epsilon(s,t)$, where $\eta(s,t)$ follows a separable spatio-temporal covariance matrix; $\epsilon(t,s)$ is i.i.d Gaussian random noise.

$B=3200$, $K=36$, $L=36$.

## Bayes Error

assume we know everything about the data generation process.

| missing rate | missing pattern | MAE  |
| ------------ | --------------- | ---- |
| 0.1          | random          | 0.80 |
| 0.1          | block           | 0.87 |
| 0.5          | random          | 0.82 |
| 0.5          | block           | 0.89 |
| 0.9          | random          | 0.87 |
| 0.9          | block           | 0.93 |



## Spatio-Temporal Kriging

Horribly slow





## Mean imputer

| missing rate | missing pattern | MAE  |
| ------------ | --------------- | ---- |
| 0.1          | random          | 1.08 |
| 0.1          | block           | 1.11 |
| 0.5          | random          | 1.07 |
| 0.5          | block           | 1.12 |
| 0.9          | random          | 1.08 |
| 0.9          | block           | 1.13 |



## Interpolation imputer

| missing rate | missing pattern | MAE  |
| ------------ | --------------- | ---- |
| 0.1          | random          | 0.99 |
| 0.1          | block           | 1.25 |
| 0.5          | random          | 1.02 |
| 0.5          | block           | 1.27 |
| 0.9          | random          | 1.20 |
| 0.9          | block           | 1.28 |



## BIRNN

| missing rate | missing pattern | MAE  |
| ------------ | --------------- | ---- |
| 0.1          | random          | 0.86 |
| 0.1          | block           | 1.04 |
| 0.5          | random          | 0.89 |
| 0.5          | block           | 1.06 |
| 0.9          | random          | 1.02 |
| 0.9          | block           | 1.06 |



## CSDI

| missing rate | missing pattern | MAE  |
| ------------ | --------------- | ---- |
| 0.1          | random          | 0.88 |
| 0.1          | block           | 1.02 |
| 0.5          | random          | 0.91 |
| 0.5          | block           | 1.04 |
| 0.9          | random          | 1.43 |
| 0.9          | block           | 1.06 |



# Visualization of CSDI

missing rate = 0.1, missing pattern = random

<img src="/Users/kehuiyao/Desktop/CSDI/figures/synthetic_ST_separable_0.1_random_CSDI.png" alt="synthetic_ST_separable_0.1_random_CSDI" style="zoom: 33%;" />

missing rate = 0.1, missing pattern = block

![synthetic_ST_separable_0.1_block_CSDI](/Users/kehuiyao/Desktop/CSDI/figures/synthetic_ST_separable_0.1_block_CSDI.png)

missing rate = 0.5, missing pattern=random

<img src="/Users/kehuiyao/Desktop/CSDI/figures/synthetic_ST_separable_0.5_random_CSDI.png" alt="synthetic_ST_separable_0.5_random_CSDI" style="zoom:72%;" />

missing rate = 0.5, missing pattern=block

<img src="/Users/kehuiyao/Desktop/CSDI/figures/synthetic_ST_separable_0.5_block_CSDI.png" alt="synthetic_ST_separable_0.5_block_CSDI" style="zoom:72%;" />



missing rate=0.9, missing pattern=random

<img src="/Users/kehuiyao/Desktop/CSDI/figures/synthetic_ST_separable_0.9_random_CSDI.png" alt="synthetic_ST_separable_0.9_random_CSDI" style="zoom:72%;" />

missing rate=0.9, missing pattern=block

<img src="/Users/kehuiyao/Desktop/CSDI/figures/synthetic_ST_separable_0.9_block_CSDI.png" alt="synthetic_ST_separable_0.9_block_CSDI" style="zoom:72%;" />
