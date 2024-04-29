# Quinton Nelson
# 4/22/2024
# This script trains a simple convolutional neural network (CNN) model for emotion recognition using the FER2013 dataset.


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
            nn.Dropout(0.25),  # Dropout layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer
            nn.Linear(256, 5)  # 5 emotions
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(11):  # Number of epochs
    for imgs, labels in dataloader:
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Log the loss at the end of each epoch
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    with open('F:/PythonProjects/EmotionRecognition/data/training_loss.txt', 'a') as f:
        f.write(f'Epoch {epoch+1}, Loss: {loss.item()}\n')

# Save model
torch.save(model.state_dict(), 'emotion_recognition_model.pth')