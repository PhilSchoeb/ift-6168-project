"""
Build dataset consisting of static grating images to eventually merge with natural scenes dataset
"""

import numpy as np

def generate_gratings(orientations,
                      spatial_freqs,
                      phases,
                      size=(128, 128),
                      contrast=1.0,
                      map_to_0_1=True):
    """
    Generate a batch of sinusoidal gratings.

    Parameters
    ----------
    orientations : array-like (N,)
    spatial_freqs : array-like (N,)
    phases : array-like (N,)
    size : (H, W)
    contrast : float or array-like (N,)
    map_to_0_1 : bool

    Returns
    -------
    imgs : (N, H, W)
    """

    orientations = np.asarray(orientations)
    spatial_freqs = np.asarray(spatial_freqs)
    phases = np.asarray(phases)

    N = len(orientations)

    # coordinate grid (shared for all images)
    x = np.linspace(-1, 1, size[1])
    y = np.linspace(-1, 1, size[0])
    X, Y = np.meshgrid(x, y)

    # allocate output
    imgs = np.empty((N, size[0], size[1]), dtype=np.float32)

    for i in range(N):
        theta = np.deg2rad(orientations[i])
        sf = spatial_freqs[i]
        phi = 2 * np.pi * phases[i]

        # rotate coordinates
        Xr = X * np.cos(theta) + Y * np.sin(theta)

        # grating
        img = contrast * np.sin(2 * np.pi * sf * Xr + phi)

        if map_to_0_1:
            img = 0.5 + 0.5 * img

        imgs[i] = img.astype(np.float32)

    return imgs