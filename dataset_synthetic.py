import numpy as np
import matplotlib.pyplot as plt

def generate_ST_data_with_separable_covariance(K, L, B, seed=42):
    """
    Generate spatio-temporal data with separable covariance function.
    y(s,t) = eta(s, t) + epsilon(s, t), where eta(s,t) are spatial-temporal random effects, and epsilon(s,t) are independent noise.

    We use Cholesky decomposition to generate eta. Note that the covariance function is separable, so C = np.kron(C_temporal, C_spatial), and by Cholesky decomposition, C_temporal = L_temporal @ L_temporal.T, C_spatial = L_spatial @ L_spatial.T.

    By the property of kronecker product, np.kron(L_temporal, L_spatial) @ np.kron(L_temporal, L_spatial).T = np.kron(L_temporal @ L_temporal.T, L_spatial @ L_spatial.T) = np.kron(C_temporal, C_spatial).


    :param K:
    :param L:
    :param B:
    :param seed:
    :return:
    """
    # define random seed generator
    rng = np.random.RandomState(seed=seed)

    # Define the spatio-temporal domain
    x = np.linspace(0, 1, K)
    t = np.linspace(0, 1, L)

    # Choose covariance functions and parameters
    # Choose covariance functions and parameters
    def gaussian_covariance(x1, x2, length_scale):
        return np.exp(-0.5 * (x1 - x2) ** 2 / length_scale ** 2)

    length_scale_space = 0.2
    length_scale_time = 0.2
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
    eta = rng.normal(0, 1, size=(len(x)*len(t),B))

    # the Cholesky decomposition of the separable covariance function is
    L_spatial_temporal = np.kron(L_temporal, L_spatial)

    # Generate correlated random variables by multiplying L and eta
    eta = L_spatial_temporal @ eta

    # reshape eta to 2D
    eta = eta.reshape(len(t), len(x), B)

    # Generate independent standard normal random variables
    epsilon = rng.normal(0, 1, size=(len(t), len(x), B))

    # y = eta + epsilon
    y = eta + epsilon

    # reshape to B, K, L
    y = y.transpose(2, 1, 0)

    # Plot the generated data
    plt.imshow(y[0, :, :], cmap='jet')
    plt.show()

    # standardize the data, record the mean and std
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_standardized = (y - y_mean) / y_std
    return y_standardized, y_mean, y_std, spatial_cov

if __name__ == "__main__":
    K = 100
    T = 50
    B = 5
    output, _, _, _ = generate_ST_data_with_separable_covariance(K, T, B, seed=42)
    print(output.shape)
