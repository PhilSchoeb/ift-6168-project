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

from data import StaticGratingsDataset

import pickle
import rfcde
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, TruncatedSVD, PCA
from sklearn.metrics.pairwise import rbf_kernel

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


def get_sklearn_kernel_density(i, j, bandwidth_i=1.0, bandwidth_j=1.0):
    num_samples = i.shape[0]
    assert j.shape[0] == num_samples
    assert len(i.shape) == 2
    assert len(j.shape) == 2

    x_weights = rbf_kernel(i, gamma=1.0/bandwidth_i**2)
    y_weights = rbf_kernel(j, gamma=1.0/bandwidth_j**2)

    # Nadaraya-Watson
    conditional = (x_weights * y_weights)
    conditional /= x_weights.sum(axis=1, keepdims=True)

    assert conditional.shape == (num_samples, num_samples)
    return conditional

def main():
    print("Getting dataset...")
    sg_dataset = StaticGratingsDataset(750332458)
    h_v_bars = sg_dataset.get_presentation_ids(orientation=[0, 90])
    visp_units = sg_dataset.get_unit_ids("VISp")
    X_sg, y_sg = sg_dataset.get_data(presentation_ids=h_v_bars, unit_ids=visp_units, stimulus_type="params")
    i = X_sg
    #TODO preprocess i to have same scale as j
    j = y_sg
    print(f"i shape: {i.shape}")
    print(f"j shape: {j.shape}")
    print("Applying dimensionality reductions to J...")
    j_reduced_svd, j_reduced_nmf, j_reduced_pca = j_dimensionality_reductions(y_sg, 20, True)
    print(f"j_reduced_pca shape: {j_reduced_pca.shape}")
    print("Calculating conditional density...")
    #density = get_density_RFCDE(i, j_reduced_pca, n_basis=5, bandwidth=0.005)
    density = get_sklearn_kernel_density(i, j_reduced_pca)
    print(type(density))
    save_name = "density.pkl"
    with open(save_name, "wb") as f:
        pickle.dump(density, f)
    print(f"Density estimation done. density object saved as {save_name}")


if __name__ == "__main__":
    main()