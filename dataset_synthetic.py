import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import multivariate_normal

from torch import nn
from nn.layers.gcn import GCN
import torch
import copy
import timesynth as ts


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
    5. other covariates: X_other(s, t)

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
            X1[:, :, i] = t[i]
        # linear in space
        X2 = np.zeros([B, K, L])
        for i in range(K):
            X2[:, i, :] = x[i]

        # linear in space and time interaction
        X3 = np.zeros([B, K, L])
        for i in range(K):
            for j in range(L):
                X3[:, i, j] = x[i] * t[j]

        # other covariates
        X4 = np.zeros([B, K, L, 50])  # 50 other covariates
        for i in range(50):
            X4[:, :, :, i] = rng.normal(0, 1, size=[B, K, L])

        # stack all basis functions
        X = np.stack([X0, X1, X2, X3], axis=3)

        # concatenate other covariates
        X = np.concatenate([X, X4], axis=3)

        # Generate random coefficients
        beta = rng.normal(0, 1, size=X.shape[3])

        # Generate f(s,t) = X(s,t)*beta(s,t)
        f = np.einsum('bklj, j->bkl', X, beta)  # (B, K, L)
        if non_linear_additive is not None:
            # generate non-linear additive effects using neural network, relu activation function
            input_dim = X.shape[3]
            hidden_dim = [32, 16]
            output_dim = 1
            W = [rng.uniform(-np.sqrt(1/input_dim), np.sqrt(1/input_dim), size=[input_dim, hidden_dim[0]]), rng.uniform(-np.sqrt(1/hidden_dim[0]), np.sqrt(1/hidden_dim[0]), size=[hidden_dim[0], hidden_dim[1]]), rng.uniform(-np.sqrt(1/hidden_dim[1]), np.sqrt(1/hidden_dim[1]), size=[hidden_dim[1], output_dim])]

            def relu(x):
                return np.maximum(x, 0)

            g = relu(np.einsum('bklj, ji->bkli', X, W[0]))
            g = relu(np.einsum('bkli, ij->bklj', g, W[1]))
            g = np.einsum('bklj, ji->bkli', g, W[2])

            g = g * 10  # scale up the non-linear effects

            f = f + g  # linear + non-linear

    else:
        f = np.zeros([B, K, L])
        X = None



    ############################### Generate eta(s,t) ##########################################
    # Choose covariance functions and parameters
    def gaussian_covariance(x1, x2, length_scale):
        return np.exp(-0.5 * (x1 - x2) ** 2 / length_scale ** 2)

    # length_scale_space = 1 / 36  # strong spatial correlation
    length_scale_space = 1e-4  # weak spatial correlation

    length_scale_time = 1 / 36  # strong temporal correlation
    # length_scale_time = 1e-4  # weak temporal correlation

    jitter = 1e-6

    # Generate the covariance matrix for spatial domain
    # outer product of x1 and x2 based on function gaussian_covariance
    spatial_cov = gaussian_covariance(
        x1=np.expand_dims(x, 1),
        x2=np.expand_dims(x, 0),
        length_scale=length_scale_space
    )

    # outer product of t1 and t2 based on function gaussian_covariance
    temporal_cov = gaussian_covariance(
        x1=np.expand_dims(t, 1),
        x2=np.expand_dims(t, 0),
        length_scale=length_scale_time
    )

    # avoid singular matrix
    spatial_cov += jitter * np.eye(len(x))
    temporal_cov += jitter * np.eye(len(t))

    # Compute Cholesky decomposition for spatial and temporal covariance matrices
    L_spatial = np.linalg.cholesky(spatial_cov)
    L_temporal = np.linalg.cholesky(temporal_cov)

    # Generate independent standard normal random variables
    eta = rng.normal(0, 1, size=(B, len(x) * len(t)))  # (B, K*L)
    # the Cholesky decomposition of the separable covariance function is
    L_spatial_temporal = np.kron(L_spatial, L_temporal)  # (K*L, K*L)

    # Generate correlated random variables by multiplying L and eta
    eta = np.einsum('ij, bj->bi', L_spatial_temporal, eta)  # (B, K*L)

    # reshape eta
    eta = eta.reshape(B, K, L)  # (B, K, L)

    ############################### Generate epsilon(s,t) ##########################################
    # Generate independent standard normal random variables
    epsilon = rng.normal(0, 1, size=(B, K, L))  # (B, K, L)

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
    C_spatio_temporal = (L_spatial_temporal @ L_spatial_temporal.T + np.eye(len(x)*len(t))) / y_std**2

    return y_standardized, y_mean, y_std, spatial_cov, C_spatio_temporal, X

def generate_ST_data_with_space_time_basis_functions(K, L, B, linear_additive=None, non_linear_additive=None, seed=42):
    """
    Generate spatio-temporal data with space-time basis functions,
    y(s,t) = f(s,t) + eta(s, t) + epsilon(s, t),
    where f(s,t) = linear_additive + non_linear_additive,
    eta(s,t) is a sum of space-time basis functions with random coefficients.

    :param K:
    :param L:
    :param B:
    :param linear_additive:
    :param non_linear_additive:
    :param seed:
    :return:
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
    num_space_basis = 5
    num_time_basis = 5
    num_space_time_basis = num_space_basis * num_time_basis
    space_basis_func_list = []
    time_basis_func_list = []
    space_time_basis_func_list = []

    # Generate space basis functions
    def space_basis_func_generator(i):
        def space_basis_func(x):
            if np.abs(x-i) < 0.2:
                return np.power(1 - np.power(np.abs(x-i) / 0.2, 2), 2)
            else:
                return 0
        return space_basis_func

    def time_basis_func_generator(i):
        def time_basis_func(t):
            if np.abs(t-i) < 0.2:
                return np.power(1 - np.power(np.abs(t-i) / 0.2, 2), 2)
            else:
                return 0
        return time_basis_func

    def space_time_basis_func_generator(i, j, space_basis_func_list, time_basis_func_list):
        def space_time_basis_func(x, t):
            return space_basis_func_list[i](x) * time_basis_func_list[j](t)
        return space_time_basis_func

    # Generate space basis functions
    for i in range(num_space_basis):
        space_basis_func_list.append(space_basis_func_generator(1 / num_space_basis * i))

    # Generate time basis functions
    for i in range(num_time_basis):
        time_basis_func_list.append(time_basis_func_generator(1 / num_time_basis * i))

    # Generate space-time basis functions
    for i in range(num_space_basis):
        for j in range(num_time_basis):
            space_time_basis_func_list.append(space_time_basis_func_generator(i, j, space_basis_func_list, time_basis_func_list))

    # create a array of (num_space_basis, K, L)
    space_basis_values = np.zeros([num_space_basis, K, L])
    for l in range(num_space_basis):
        for i in range(K):
            for j in range(L):
                space_basis_values[l, i, j] = space_basis_func_list[l](x[i])

    # create a array of (num_time_basis, K, L)
    time_basis_values = np.zeros([num_time_basis, K, L])
    for l in range(num_time_basis):
        for i in range(K):
            for j in range(L):
                time_basis_values[l, i, j] = time_basis_func_list[l](t[j])

    # create a array of (num_space_time_basis, K, L)
    space_time_basis_values = np.zeros([num_space_time_basis, K, L])
    for l in range(num_space_time_basis):
        for i in range(K):
            for j in range(L):
                space_time_basis_values[l, i, j] = space_time_basis_func_list[l](x[i], t[j])

    # concatenate space, time, and space-time basis values to get a array of (num_space_basis + num_time_basis + num_space_time_basis, K, L)
    all_basis_values = np.concatenate([space_basis_values, time_basis_values, space_time_basis_values], axis=0)

    # Generate random coefficients with size (B, num_space_basis + num_time_basis + num_space_time_basis
    eta_coefficients = rng.normal(0, 1, size=(B, num_space_basis + num_time_basis + num_space_time_basis))

    # Generate eta(s,t) = sum_{l=1}^{num_space_basis+num_time_basis+num_space_time_basis} eta_coefficients[l] * all_basis_values[l]
    eta = np.einsum('bn, nkl->bkl', eta_coefficients, all_basis_values)  # (B, K, L)


    ############################### Generate epsilon(s,t) ##########################################
    # Generate independent standard normal random variables
    epsilon = rng.normal(0, 1, size=(B, K, L))  # (B, K, L)

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
    C_spatio_temporal = None
    spatial_cov = None

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


def generate_ST_data_with_dynamic_process_model(K, L, B, seed=42):
    """
    Generate spatio-temporal data with dynamic process model
    Y(s, t) = g(Z(s, t)) + epsilon(s, t)
    Z(s, t) = f(Z(:, t-1)) + eta(s, t)
    :param K:
    :param L:
    :param B:
    :param seed:
    :return:
    """
    # define the latent process model using RNN and GCN
    hidden_dim = 64
    rnn = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)  # define the RNN

    gcn = GCN(input_size=hidden_dim, hidden_size=hidden_dim, output_size=hidden_dim)  # define the GCN

    # define the neural network that maps the latent process to the observed process
    mlp = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.LayerNorm([hidden_dim])
    )

    latent_to_observed = nn.ModuleList([copy.deepcopy(mlp) for _ in range(5)] +
                                        [nn.Linear(hidden_dim, K)])


    # generate an adjacency matrix
    adjacency_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            adjacency_matrix[i, j] = np.exp(-np.abs(i - j) / 5)

    adjacency_matrix = torch.from_numpy(adjacency_matrix).float()

    # Z is the latent process, intialized as standard normal
    Z = torch.randn(B, hidden_dim, L)  # (B, hidden_dim, L)

    # pass the input through the RNN and GCN
    rnn.eval()
    gcn.eval()
    latent_to_observed.eval()

    for i in range(L):
        if i == 0:
            pass
        else:
            tmp = rnn(Z[:, :, i-1])[0]  # (B, hidden_dim)
            Z[:, :, i] = (tmp - torch.mean(tmp)) / torch.std(tmp)  # normalize the latent process

    # generate the epsilon using torch
    epsilon = torch.randn(B, K, L)

    # generate the observation
    Z = Z.permute(0, 2, 1)  # (B, L, hidden_dim)

    for layer in latent_to_observed:
        Z = layer(Z)

    y = Z  # (B, L, K)
    # normalize the observation
    y = (y - torch.mean(y)) / torch.std(y)
    y = y.permute(0, 2, 1)  # (B, K, L)
    y = y + epsilon  # (B, K, L)

    # convert to numpy array
    y = y.detach().numpy()

    # Plot the generated data
    plt.imshow(y[0, :, :], cmap='jet')
    plt.show()

    # standardize the data, record the mean and std
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_standardized = (y - y_mean) / y_std

    return y_standardized, y_mean, y_std

def generate_ST_data_with_complex_time_seires_model(K, L, B, seed=42):
    time_sampler = ts.TimeSampler(stop_time=35)
    regular_time_samples = time_sampler.sample_regular_time(num_points=36)
    sinusoid = ts.signals.Sinusoidal(frequency=0.25)
    white_noise = ts.noise.GaussianNoise(std=0.3)
    timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)

    y = np.zeros((B, K, L))
    for b in range(B):
        for k in range(K):
            samples, _, _ = timeseries.sample(regular_time_samples)
            y[b, k, :] = samples

    # Plot the generated data
    plt.imshow(y[0, :, :], cmap='jet')
    plt.show()

    # standardize the data, record the mean and std
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_standardized = (y - y_mean) / y_std

    return y_standardized, y_mean, y_std



if __name__ == "__main__":
    K = 36
    T = 36
    B = 1
    # output, *_ = generate_ST_data_with_separable_covariance(K, T, B, seed=42)
    # output, *_ = generate_ST_data_with_separable_covariance(K, T, B, seed=42, linear_additive=True, non_linear_additive=True)
    # output, *_ = generate_ST_data_with_space_time_basis_functions(K, T, B, linear_additive=None, non_linear_additive=None, seed=42)
    # output, *_ = generate_ST_data_with_dynamic_process_model(K, T, B, seed=42)
    output, *_ = generate_ST_data_with_complex_time_seires_model(K, T, B, seed=42)
    print(output.shape)


