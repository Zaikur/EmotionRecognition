# Quinton Nelson
# 4/14/2024
# This script loads the data that has already been processed and saved to .npy files for use or training the model.
# Before returning the data, the function converts the numpy arrays to Torch tensors to be used in PyTorch models.

import os
import numpy as np
import torch


def load_npy_data(data_type='train'):
    data_path = f'data/processed/{data_type}'
    images = np.load(os.path.join(data_path, 'images.npy'), allow_pickle=True)
    labels = np.load(os.path.join(data_path, 'labels.npy'))

    # Ensure images are numpy arrays (correct the dtype if necessary)
    if images.dtype == object:  # If images are in an object array (e.g., array of arrays)
        images = np.stack(images)  # This will form a single numpy array from a list of arrays

    # Convert numpy arrays to Torch tensors
    images = torch.from_numpy(images.astype(np.float32))  # Ensure images are float32
    labels = torch.tensor(labels, dtype=torch.int64)  # Labels should be int64 for PyTorch

    return images, labels


