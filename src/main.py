# Quinton Nelson
# 4/22/2024
# Main script for running the emotion recognition model on a live video stream from a webcam.


import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from models.emotion_model import EmotionClassifier

# Emotion labels dictionary
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprise"}

# Load the trained model
model = EmotionClassifier()
model.load_state_dict(torch.load('emotion_recognition_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load a pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from a webcam
cap = cv2.VideoCapture(0)

# Define transforms for the face images
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    # Scale factor - Parameter specifying how much the image size is reduced at each image scale.
    # Min neighbors - Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    # Min size - Minimum possible object size. Objects smaller than this are ignored.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw rectangles around each face and predict emotions
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))  # Resize to the size model expects
        face_img = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # Normalize
        face_img = (face_img - 0.485) / 0.229  # Use the same normalization as during training

        with torch.no_grad():
            outputs = model(face_img)
            probabilities = F.softmax(outputs, dim=1)
            max_prob, predictions = torch.max(probabilities, 1)
            predicted_emotion = emotion_labels[predictions.item()]
            confidence = max_prob.item() * 100  # percentage

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        emotion_text = f"{predicted_emotion}"
        
        # Position the emotion text above the box
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        # Position the confidence text below the box
        cv2.putText(frame, f"{confidence:.2f}%", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
