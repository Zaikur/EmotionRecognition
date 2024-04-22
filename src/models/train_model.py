# Quinton Nelson
# 4/22/2024
# This script trains a convolutional neural network model for emotion recognition using PyTorch. 
# The script loads the preprocessed data from .npy files, creates a simple CNN model, defines the loss function and optimizer, 
# and trains the model for a specified number of epochs. The trained model is then saved to a file for future use. 

import sys
sys.path.insert(1, 'src/data')
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from load_npy_data import load_npy_data

# Load the dataset
images, labels = load_npy_data('train')

# Create a dataset and dataloader
dataset = TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define a CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # 7 emotions
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Number of epochs
    for imgs, labels in dataloader:
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Save model checkpoint to file, overwriting existing file
    with open('F:/PythonProjects/EmotionRecognition/data/training_loss.txt', 'w') as f:
        for epoch in range(10):  # Number of epochs
            for imgs, labels in dataloader:
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            f.write(f'Epoch {epoch+1}, Loss: {loss.item()}\n')

    
    # Save model checkpoint to file
    with open('F:/PythonProjects/EmotionRecognition/data/training_loss.txt', 'a') as f:
        f.write(f'Epoch {epoch+1}, Loss: {loss.item()}\n')

# Save model
torch.save(model.state_dict(), 'emotion_recognition_model.pth')
