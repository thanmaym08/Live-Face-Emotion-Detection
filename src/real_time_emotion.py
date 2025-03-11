import cv2
import dlib
import numpy as np
from keras.models import load_model

# Load pre-trained face detector (dlib)
detector = dlib.get_frontal_face_detector()

# Load pre-trained emotion classification model (modify the path accordingly)
emotion_model = load_model("models/emotion_model.h5")

# Emotion labels based on FER-2013 dataset
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = detector(gray)  # Detect faces

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Extract the face ROI from the original color frame (not grayscale)
        face_roi = frame[y:y+h, x:x+w]



# Resize to 224x224 for MobileNetV2
        face_resized = cv2.resize(face_roi, (224, 224))

# Normalize pixel values (convert to float range 0-1)
        face_normalized = face_resized / 255.0

# Reshape for MobileNetV2 (batch size, height, width, channels)
        face_reshaped = np.reshape(face_normalized, (1, 224, 224, 3))


        # Predict emotion
        prediction = emotion_model.predict(face_reshaped)
        emotion_label = emotion_labels[np.argmax(prediction)]  # Get highest probability emotion

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display emotion label
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
