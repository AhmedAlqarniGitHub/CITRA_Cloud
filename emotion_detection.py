import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1: INFO, 2: WARNING, and 3: ERROR)
import tensorflow as tf
from keras.models import load_model

# Dictionary containing the emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}

# Function to get the labels
def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return emotion_dict
    raise ValueError("Dataset not recognized. Please provide a valid dataset name.")

# Load the pre-trained model and its labels
emotion_model_path = 'best_model.h5'
emotion_labels = get_labels('fer2013')

# Load the model
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

# Directory containing the cropped face images
face_images_dir = '../face_detection/detected_faces'

# Function to preprocess the face image
def preprocess_face(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, emotion_target_size)
    face_image = face_image.astype('float32') / 255.0
    face_image = np.expand_dims(face_image, 0)
    face_image = np.expand_dims(face_image, -1)
    return face_image

# Process each image in the directory
for filename in os.listdir(face_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Load the cropped face image
        face_image = cv2.imread(os.path.join(face_images_dir, filename))

        # Preprocess the face image
        face_image = preprocess_face(face_image)

        # Predict the emotion
        emotion_prediction = emotion_classifier.predict(face_image)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

        print(f"Detected Emotion for {filename}: {emotion_text} (probability: {emotion_probability:.2f})")
