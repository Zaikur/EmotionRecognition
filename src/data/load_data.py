# Quinton Nelson
# 4/14/2024
# This script loads the data for the model, preprocessing data, and saving the data to .npy files for faster loading in the future. 
# This script is used to load the images and labels from the raw data directory, preprocess the images, and save the processed data to .npy files. 
# The processed data can then be loaded directly from the .npy files for training the model, which is faster than loading the raw images and preprocessing them each time the model is trained. 
# The script also includes data augmentation techniques such as resizing, converting to grayscale, and normalizing the images before saving them to .npy files.

import os
from PIL import Image
import numpy as np
from torchvision import transforms

def load_images_from_category(data_path, category, transform, label_map):
    category_path = os.path.join(data_path, category)
    images = []
    labels = []

    for img_file in os.listdir(category_path):
        img_path = os.path.join(category_path, img_file)
        img = Image.open(img_path).convert('L')  # Convert images to grayscale
        img = transform(img)
        images.append(img)
        labels.append(label_map[category])  # Use category name as label

    return images, labels

def load_and_save_data(data_type='train'):
    data_path = f'data/raw/{data_type}'
    categories = os.listdir(data_path)
    
    # Create a label map to map category names to integer labels
    label_map = {category: i for i, category in enumerate(categories)}

    transform = transforms.Compose([
        transforms.Resize((48, 48)),  # Resize all images to 48x48
        transforms.ToTensor(),        # Convert images to tensor
        transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize the images
    ])

    all_images = []
    all_labels = []

    for category in categories:
        images, labels = load_images_from_category(data_path, category, transform, label_map)
        all_images.extend(images)
        all_labels.extend(labels)

    all_images = np.array(all_images, dtype='object')
    all_labels = np.array(all_labels, dtype='int')

    save_data_to_npy(all_images, all_labels, data_type)

def save_data_to_npy(images, labels, data_type='train'):
    save_path = f'data/processed/{data_type}'
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    np.save(os.path.join(save_path, 'images.npy'), images)  # Save images
    np.save(os.path.join(save_path, 'labels.npy'), labels)  # Save labels

# Load and save images and labels for training data
load_and_save_data('train')
