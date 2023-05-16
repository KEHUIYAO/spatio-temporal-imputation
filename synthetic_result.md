# Spatio-Temporal Model

Model: $Y(s,t) = f(s,t) + \eta(s, t)+\epsilon(s,t)$, where $f(s,t)$ is space-time fixed effect, $\eta(s,t)$ is space-time random effect, $\epsilon(t,s)$ is i.i.d Gaussian random noise.

## $f(s,t)$

### Linear



### Non-linear



## $\eta(s,t)$

### A separable spatio-temporal covariance function

First, we assume $\eta(s,t)$ follow a separable spatio-temporal covariance matrix, where

$C{(s_1,t_1),(s_2,t_2)}=C_s(s_1,s_2)\times C_t(t_1,t_2)$,

$C_s(s_1, s_2) = \exp[-0.5\times \{d(s_1,s_2) / \sigma_s\}^2]$,

$C_t(t_1, t_2) = \exp[-0.5\times \{(t_1-t_2) / \sigma_t\}^2]$.

### Random effects with spatial basis functions

$\eta(s,t)=\sum_{i=1}^{n_{\alpha}}\phi_i(s)\alpha_i(t)$, where

$\phi_i(s)$ are known basis functions, and $\alpha_i(t)$ can be a temporal random process.

### Random effects with temporal basis functions

$\eta(s,t)=\sum_{i=1}^{n_{\alpha}}\phi_i(s)\alpha_i(s)$.





# Missing Patterns

## Randomly missing



## Block missing









# Methods

## Mean imputer



## Interpolation



## Spatio-Temporal Kriging



## Bidirectional RNN 



## Bidirectional RNN + GCN



## Diffusion-based Bidirectional RNN + GCN







# Simulation

## Setting 1: weak temporal and strong spatial correlation

Let $s$ be $K$ evenly spaced points between 0 and 1, $t$ be $L$ evenly spaced points between 0 and 1.

Set $\sigma_s=1/K$, $\sigma_t=1e^{-4}$. 

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

