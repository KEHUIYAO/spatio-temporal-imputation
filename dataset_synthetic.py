import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import multivariate_normal


def generate_ST_data_with_separable_covariance(K, L, B, linear_additive=None, non_linear_additive=None, seed=42):
    """
    Generate spatio-temporal data with y(s,t) = f(s,t) + eta(s, t) + epsilon(s, t),
    where f(s,t) = linear_additive + non_linear_additive,
    eta(s,t) are spatial-temporal random effects with spatio-temporal separable covariance function,
    epsilon(s,t) are independent noise.

    If f(s, t) are pure linear additive effects, say, f(s,t)=X(s,t)*beta(s,t). We can generate X(s, t) by incorporating some spatial basis functions and temporal basis functions in the model and randomly draw coefficents beta(s,t) from a multivariate normal distribution.

    The basis function we include are:
    1. overall mean: X0(s, t)=1
    2. linear in space: X1(si, t)=si for all t
    3. linear in time: X2(s, tj)=tj for all s
    Optional:
    4. linear in space and time interaction: X3(si, tj)=si*tj

    If f(s, t) contains both linear additive effects or non-linear additive effects, say, f(s,t)=X(s,t)*beta(s,t) + g(s,t), where g(s,t) is a non-linear function. We can generate X(s, t) by incorporating some spatial basis functions and temporal basis functions in the model and randomly draw coefficents beta(s,t) from a multivariate normal distribution. We can generate g(s,t) ... (to be continued)

    To efficiently generate eta(s, t), we use Cholesky decomposition on the spatio-temporal covariance matrix. Note that the covariance function is separable, where C = np.kron(C_temporal, C_spatial), and by Cholesky decomposition, C_temporal = L_temporal @ L_temporal.T, C_spatial = L_spatial @ L_spatial.T and by the property of kronecker product, np.kron(L_temporal, L_spatial) @ np.kron(L_temporal, L_spatial).T = np.kron(L_temporal @ L_temporal.T, L_spatial @ L_spatial.T) = np.kron(C_temporal, C_spatial). Finally, we only need to generate independent standard normal random variables and multiply it with np.kron(L_temporal, L_spatial) to get eta(s, t).


    """
    # define random seed generator
    rng = np.random.RandomState(seed=seed)

    # Define the spatio-temporal domain
    x = np.linspace(0, 1, K)
    t = np.linspace(0, 1, L)
    ############################### Generate f(s,t) ##########################################
    if linear_additive is not None:
        # overall mean
        X0 = np.ones([B, K, L])
        # linear in time
        X1 = np.zeros([B, K, L])
        for i in range(L):
            X1[:, :, i] = t[i] * 20
        # linear in space
        X2 = np.zeros([B, K, L])
        for i in range(K):
            X2[:, i, :] = x[i] * 10

        # basis functions (B, K, L, 3)
        X = np.stack([X0, X1, X2], axis=3)

        # Generate random coefficients
        beta = rng.normal(0, 1, size=3)  # 3

        # Generate f(s,t) = X(s,t)*beta(s,t)
        f = np.einsum('bklj, j->bkl', X, beta)  # (B, K, L)
    elif non_linear_additive is not None:
        f = np.zeros([B, K, L])
    else:
        f = np.zeros([B, K, L])



    ############################### Generate eta(s,t) ##########################################
    # Choose covariance functions and parameters
    def gaussian_covariance(x1, x2, length_scale):
        return np.exp(-0.5 * (x1 - x2) ** 2 / length_scale ** 2)

    length_scale_space = rng.uniform(1/36, 1, size=B)
    length_scale_time = rng.uniform(1/36, 1, size=B)

    jitter = 1e-6

    # Generate the covariance matrix for spatial domain
    # outer product of x1 and x2 based on function gaussian_covariance
    spatial_cov = [gaussian_covariance(
        x1=np.expand_dims(x, 1),
        x2=np.expand_dims(x, 0),
        length_scale=lengthScaleSpace
    ) + jitter * np.eye(len(x)) for lengthScaleSpace in length_scale_space]  # a list of B (K, K) arrays

    # outer product of t1 and t2 based on function gaussian_covariance
    temporal_cov = [gaussian_covariance(
        x1=np.expand_dims(t, 1),
        x2=np.expand_dims(t, 0),
        length_scale=lengthScaleTime
    ) + jitter * np.eye(len(t)) for lengthScaleTime in length_scale_time]  # a list of B (L, L) arrays


    # Compute Cholesky decomposition for spatial and temporal covariance matrices
    L_spatial = [np.linalg.cholesky(spatialCov) for spatialCov in spatial_cov]  # a list of B (K, K) arrays
    L_temporal = [np.linalg.cholesky(temporalCov) for temporalCov in temporal_cov]  # a list of B (L, L) arrays

    # Generate independent standard normal random variables
    eta = [rng.normal(0, 1, size=(len(x) * len(t))) for _ in range(B)]  # a list of B (K*L)

    # the Cholesky decomposition of the separable covariance function is
    L_spatial_temporal = [np.kron(LSpatial, LTemporal) for LSpatial, LTemporal in zip(L_spatial, L_temporal)] # a list of B (K*L, K*L) arrays


    # Generate correlated random variables by multiplying L and eta
    eta = [np.einsum('ij, j->i', LSpatialTemporal, Eta) for LSpatialTemporal, Eta in zip(L_spatial_temporal, eta)]  # (B, K*L)

    eta = np.stack(eta, axis=0)  # (B, K*L)

    # reshape eta
    eta = eta.reshape(B, K, L)  # (B, K, L)

    ############################### Generate epsilon(s,t) ##########################################
    # Generate independent standard normal random variable
    noise_std = rng.uniform(0, 2, size=B)  # (B)
    epsilon = [rng.normal(0, noiseStd, size=(K, L)) for noiseStd in noise_std]  # a list of B (K, L) arrays
    epsilon = np.stack(epsilon, axis=0)  # (B, K, L)

    ############################### Generate y(s,t) ##########################################
    # y = f + eta + epsilon
    y = f + eta + epsilon

    # Plot the generated data
    plt.imshow(y[0, :, :], cmap='jet')
    plt.show()

    # standardize the data, record the mean and std
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_standardized = (y - y_mean) / y_std

    # spatio-temporal covariance matrix of y
    C_spatio_temporal = [LSpatialTemporal@LSpatialTemporal.T+ np.eye(len(x)*len(t)) / y_std**2 for LSpatialTemporal in L_spatial_temporal]


    # covariates time
    X_time = np.zeros([B, K, L])
    for i in range(L):
        X_time[:, :, i] = t[i] * 20

    # covariates space
    X_space = np.zeros([B, K, L])
    for i in range(K):
        X_space[:, i, :] = x[i] * 10

    X = np.stack([X_time, X_space], axis=3)  # (B, K, L, 2)


    return y_standardized, y_mean, y_std, spatial_cov, C_spatio_temporal, X


if __name__ == "__main__":
    K = 36
    T = 36
    B = 200
    output, *_ = generate_ST_data_with_separable_covariance(K, T, B, seed=42)
    # output, *_ = generate_ST_data_with_separable_covariance(K, T, B, seed=42, linear_additive=True)
    print(output.shape)

