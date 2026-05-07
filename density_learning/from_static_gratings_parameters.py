"""
Assuming I of shape (num_static_gratings, 3) where second dimension represents parameters
1. orientation
2. spatial_frequency
3. phase
And assuming J of shape (num_static_gratings, num_bins, num_neurons)
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)

from data import StaticGratingsDataset

import numpy as np
import pickle
import rfcde
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, TruncatedSVD, PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

def j_dimensionality_reductions(j, n_components, return_explained_var=False):
    assert len(j.shape) == 3, f"j shape has to be (num_static_gratings, num_bins, num_neurons)"
    num_samples = j.shape[0]
    j_flat = j.reshape(num_samples, -1)
    j_sparse = csr_matrix(j_flat)

    j_reduced_svd, explained_var_svd = get_TSVD_reduction(j_sparse, n_components, return_explained_var)

    # No simple way to obtain the explained variance score with NMF
    j_reduced_nmf = get_NMF_reduction(j_flat, n_components)

    j_reduced_pca, explained_var_pca = get_PCA_reduction(j_flat, n_components, return_explained_var)

    if explained_var_svd is not None:
        print(f"TruncatedSVD explained variance: {explained_var_svd}")
    if explained_var_pca is not None:
        print(f"PCA explained variance: {explained_var_pca}")

    return j_reduced_svd, j_reduced_nmf, j_reduced_pca


def get_TSVD_reduction(data, n_components, return_explained_var=False):
    assert len(data.shape) == 2, f"data shape has to be (num_samples, num_features)"
    svd = TruncatedSVD(n_components=n_components)
    data_reduc = svd.fit_transform(data)
    explained_var = svd.explained_variance_ratio_
    total_explained_var = sum(explained_var)
    if return_explained_var:
        return data_reduc, total_explained_var
    else:
        return data_reduc, None


def get_NMF_reduction(data, n_components):
    assert len(data.shape) == 2, f"data shape has to be (num_samples, num_features)"
    nmf = NMF(n_components=n_components, max_iter=600)
    data_reduc = nmf.fit_transform(data)
    return data_reduc


def get_PCA_reduction(data, n_components, return_explained_var=False):
    assert len(data.shape) == 2, f"data shape has to be (num_samples, num_features)"
    pca = PCA(n_components=n_components)
    data_reduc = pca.fit_transform(data)
    explained_var = pca.explained_variance_ratio_
    total_explained_var = sum(explained_var)
    if return_explained_var:
        return data_reduc, total_explained_var
    else:
        return data_reduc, None


def get_density_RFCDE(i, j, n_trees=100, mtry=2, node_size=10, n_basis=None, bandwidth=None):
    """
    All arguments set as None because we do not know their default values
    """
    params = {
        "n_trees": n_trees,
        "mtry": mtry,
        "node_size": node_size,
        "n_basis": n_basis,
    }

    # remove None values
    params = {k: v for k, v in params.items() if v is not None}

    forest = rfcde.RFCDE(**params)
    forest.train(i, j)

    if bandwidth is None:
        density = forest.predict(i, j)
    else:
        density = forest.predict(i, j, bandwidth)
    return density


def get_density_fRFCDE(i, j, n_trees=100, mtry=2, node_size=10, n_basis=None, bandwidth=None, lambda_param=10):
    """
    All arguments set as None because we do not know their default values. 'lambda_param' is the exception because not
    using this argument leads to the normal RFCDE usage.
    """
    params = {
        "n_trees": n_trees,
        "mtry": mtry,
        "node_size": node_size,
        "n_basis": n_basis,
    }

    # remove None values
    params = {k: v for k, v in params.items() if v is not None}

    f_forest = rfcde.RFCDE(**params)
    f_forest.train(i, j, flambda=lambda_param)

    if bandwidth is None:
        density = f_forest.predict(i, j)
    else:
        density = f_forest.predict(i, j, bandwidth)
    return density


def get_nadaraya_watson_density(i, j, bandwidth_i=1.0, bandwidth_j=1.0):
    """
    Estimate P(J=j_b | I=i_a) for all pairs (a, b).

    Args:
        i: (num_samples, n_i)
        j: (num_samples, n_j)
        bandwidth_i: float value for rbf bandwidth on i
        bandwidth_j: float value for rbf bandwidth on j

    Returns:
        P: (num_samples, num_samples) where P[a, b] = P(J=j_b | I=i_a)
    """
    num_samples = i.shape[0]
    assert j.shape[0] == num_samples
    assert len(i.shape) == 2
    assert len(j.shape) == 2
    gamma_i = 1.0 / bandwidth_i**2
    gamma_j = 1.0 / bandwidth_j**2

    # K_i[a, s] = K(i_a, i_s) — shape (num_samples, num_samples)
    K_i = rbf_kernel(i, i, gamma=gamma_i)

    # K_j[b, s] = K(j_b, j_s) — shape (num_samples, num_samples)
    K_j = rbf_kernel(j, j, gamma=gamma_j)

    assert K_i.shape == (num_samples, num_samples)
    assert K_j.shape == (num_samples, num_samples)

    # joint[a, b] = Σ_s K(i_a, i_s) * K(j_b, j_s)
    # = K_i[a, :] @ K_j[b, :].T  =>  K_i @ K_j.T
    joint = K_i @ K_j.T  # (num_samples, num_samples)

    # marginal[a] = Σ_s K(i_a, i_s)
    marginal = K_i.sum(axis=1)  # (num_samples,)

    # P[a, b] = joint[a, b] / marginal[a]
    conditional = joint / (marginal[:, None] + 1e-10)
    assert conditional.shape == (num_samples, num_samples)

    # Transpose so first dim: change j & second dim: change i
    return np.transpose(conditional)


def get_sklearn_kernel_density(i, j, bandwidth_joint=1.0, bandwidth_i=1.0, algorithm="auto", kernel="gaussian"):
    """
    Computes the density of the joint on i and j, then divides by density on i according to:
    P(J=j|man(I=i)) = P(J=j, man(I=i)) / P(man(I=i))
    """
    kd_joint = KernelDensity(bandwidth=bandwidth_joint, algorithm=algorithm, kernel=kernel)
    kd_i = KernelDensity(bandwidth=bandwidth_i, algorithm=algorithm, kernel=kernel)

    num_samples = i.shape[0]
    num_features_i = i.shape[1]
    num_features_j = j.shape[1]
    joint = np.concatenate([i, j], axis=1)
    complete_joint = np.concatenate(
        [
            np.repeat(i, num_samples, axis=0),
            np.tile(j, (num_samples, 1))
        ],
        axis=1
    )
    assert joint.shape == (num_samples, num_features_i + num_features_j)
    assert complete_joint.shape == (num_samples**2, num_features_i + num_features_j)

    # Fit kernel density estimation routines
    kd_joint.fit(joint)
    kd_i.fit(i)

    # Retrieve log probabilities from kernel density on complete_joint
    log_prob_joint = kd_joint.score_samples(complete_joint)
    log_prob_i = kd_i.score_samples(i)

    log_prob_joint = log_prob_joint.reshape(num_samples, num_samples)
    log_conditional = log_prob_joint - log_prob_i[:, None]

    conditional = np.exp(log_conditional)
    assert conditional.shape == (num_samples, num_samples)
    print(f" Max conditional value (prenorm): {np.max(conditional)}, min conditional value (prenorm): "
          f"{np.min(conditional)}")

    # Normalization
    sum_across_js = np.sum(conditional, axis=1)[:, None]
    assert sum_across_js.shape == (num_samples, 1)
    conditional = conditional / sum_across_js
    assert conditional.shape == (num_samples, num_samples)
    print(f" Max conditional value (postnorm): {np.max(conditional)}, min conditional value (postnorm): "
          f"{np.min(conditional)}")

    return conditional


def standardize_data(data, dims=None):
    assert len(data.shape) == 2
    # If no dims provided, all dims are standardized
    if dims is None:
        len_dims = data.shape[1]
        dims = range(len_dims)

    sc = StandardScaler()
    data[:, dims] = sc.fit_transform(data[:, dims])
    return data


def main():
    print("Getting dataset...")
    sg_dataset = StaticGratingsDataset(750332458)
    h_v_bars = sg_dataset.get_presentation_ids(orientation=[0, 90])
    visp_units = sg_dataset.get_unit_ids("VISp")
    X_sg, y_sg = sg_dataset.get_data(presentation_ids=h_v_bars, unit_ids=visp_units, stimulus_type="params")
    i = X_sg
    j = y_sg
    print(f"i shape: {i.shape}")
    print(f"j shape: {j.shape}")

    print("Applying dimensionality reductions to J...")
    j_reduced_svd, j_reduced_nmf, j_reduced_pca = j_dimensionality_reductions(j, 20, True)
    print(f"j_reduced_pca shape: {j_reduced_pca.shape}")

    print(f"Standardizing data...")
    i = standardize_data(i)
    j_reduced_pca = standardize_data(j_reduced_pca)
    print(f"i means: {np.mean(i, axis=0)}")
    print(f"i stds: {np.std(i, axis=0)}")
    print(f"j means: {np.mean(j_reduced_pca, axis=0)}")
    print(f"j stds: {np.std(j_reduced_pca, axis=0)}")

    print("Calculating conditional density...")
    num_features_i = i.shape[1]
    num_features_j = j_reduced_pca.shape[1]
    bandwidth_i = 1.0 / float(num_features_i)
    bandwidth_j = 1.0 / float(num_features_j)
    bandwidth_joint = 1.0 / (float(num_features_i + num_features_j))

    # FIRST METHOD (CRASHES) ###############################################################
    #density = get_density_RFCDE(i, j_reduced_pca, n_basis=5, bandwidth=0.005)

    # SECOND METHOD (WORKS) ###############################################################
    first_bandwidth = bandwidth_i
    second_bandwidth = bandwidth_j
    density = get_nadaraya_watson_density(i, j_reduced_pca, first_bandwidth, second_bandwidth)

    # THIRD METHOD (FAILS) ################################################################
    #first_bandwidth = bandwidth_joint
    #second_bandwidth = bandwidth_i
    #density = get_sklearn_kernel_density(i, j_reduced_pca, first_bandwidth, second_bandwidth)

    print(type(density))
    save_name = "density.pkl"
    save_path = os.path.join(FILE_PATH, "out", save_name)
    with open(save_path, "wb") as f:
        pickle.dump(density, f)
    print(f"Density estimation done with bandwidths = {first_bandwidth}, {second_bandwidth}. density object"
          f" saved as {save_path}")


if __name__ == "__main__":
    main()