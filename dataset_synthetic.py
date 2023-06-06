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
    if linear_additive is not None and non_linear_additive is None:
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
    elif linear_additive is not None and non_linear_additive is not None:
        f = np.zeros([B, K, L])
        X = None
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






if __name__ == "__main__":
    K = 36
    T = 36
    B = 1
    # output, *_ = generate_ST_data_with_separable_covariance(K, T, B, seed=42)
    output, *_ = generate_ST_data_with_separable_covariance(K, T, B, seed=42, linear_additive=True)
    # output, *_ = generate_ST_data_with_space_time_basis_functions(K, T, B, linear_additive=None, non_linear_additive=None, seed=42)
    print(output.shape)


