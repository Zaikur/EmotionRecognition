# Quinton Nelson
# 4/14/2024
# This script loads the data for the model and contains simple steps such as loading the dataset from the CSV files, converting pixel strings to image arrays, and normalizing the data.

import pandas as pd
import numpy as np
import os

def load_and_process_data(data_type='train'):
    # Path to your raw data
    data_path = f'../../data/raw/{data_type}/fer2013.csv'
    data = pd.read_csv(data_path)
    pixels = data['pixels'].tolist()
    images = np.empty((len(pixels), 48, 48, 1))
    labels = data['emotion'].values

    for i, pixel_sequence in enumerate(pixels):
        single_image = np.reshape(np.array(pixel_sequence.split(), dtype="float32"), (48, 48))
        images[i, :, :, 0] = single_image

    # Normalize the images
    images /= 255.0

    return images, labels
