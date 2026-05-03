"""
Test for the Autoencoder used for dimensionality reduction of J
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FILE_PATH = os.path.dirname(__file__)

from data import StaticGratingsDataset
from dimensionality_reduction import AutoEncoder2DConv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


def main():
    print("Getting dataset...")
    sg_dataset = StaticGratingsDataset(750332458)
    h_v_bars = sg_dataset.get_presentation_ids(orientation=[0, 90])
    visp_units = sg_dataset.get_unit_ids("VISp")
    X_sg, y_sg = sg_dataset.get_data(presentation_ids=h_v_bars, unit_ids=visp_units, stimulus_type="params")
    i = X_sg
    j = y_sg

    orig_shape = j.shape
    j_flat = j.reshape(j.shape[0], -1)
    min_max_scaler = MinMaxScaler()
    j_scaled = min_max_scaler.fit_transform(j_flat)
    j = j_scaled.reshape(orig_shape)

    i_tensor = torch.tensor(i)
    j_tensor = torch.tensor(j)
    print(f"i shape: {i_tensor.shape}")
    print(f"j shape: {j_tensor.shape}")

    input_shape = j_tensor.unsqueeze(1).shape[1:]
    print(f"input_shape: {input_shape}")
    latent_dim = 20

    auto_encoder = AutoEncoder2DConv(input_shape, latent_dim)
    j_tensor = j_tensor.unsqueeze(1).float()

    x_rebuilt = auto_encoder.forward_debug(j_tensor)
    print(f"x_rebuilt shape: {x_rebuilt.shape}")


if __name__ == "__main__":
    main()