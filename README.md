# Emotion Recognition using CNN with PyTorch

## Project Overview
This project develops a convolutional neural network (CNN) to recognize human emotions from facial expressions captured through a webcam. The model is trained on the FER-2013 dataset, which consists of grayscale images labeled across seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features
- Real-time emotion recognition.
- Training with PyTorch on preprocessed image data.
- Use of data augmentation to enhance model robustness.
- Live detection and classification displayed with the predicted emotion and confidence level.

## Requirements
To run this project, you will need the following:
- Python 3.8 or above
- PyTorch 1.7 or above
- OpenCV-Python
- NumPy
- Matplotlib (for plotting training loss)

## Installation
### Clone the repository to your local machine:
git clone https://github.com/Zaikur/EmotionRecognition.git
cd emotion-recognition

### Install the required Python Packages
pip install -r requirements.txt

## Usage
- To build the required .npy files from images collected run: python src/data/load_data.py
- To start the training process, run: python src/models/train_model.py
- To execute the real-time emotion recognition, run: python src/main.py
- To plot the training loss over epochs run: python src/utilities/EpochPlotter.py

## Acknowledgments
- Thanks to the creators of the FER-2013 dataset for providing the data used in this project
- Thanks to the PyTorch team for the comprehensive documentation, and community support