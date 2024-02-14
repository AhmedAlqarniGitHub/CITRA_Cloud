import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1: INFO, 2: WARNING, and 3: ERROR)
import tensorflow as tf
from keras.models import load_model

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return emotion_dict
    raise ValueError("Dataset not recognized. Please provide a valid dataset name.")

emotion_model_path = 'fer2013_mini_XCEPTION.99-0.65.hdf5'
emotion_labels = get_labels('fer2013')
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

base_directory = 'test'  # The root directory as shown in the screenshot

def preprocess_face(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, emotion_target_size)
    face_image = face_image.astype('float32') / 255.0
    face_image = np.expand_dims(face_image, 0)
    face_image = np.expand_dims(face_image, -1)
    return face_image

# Dictionaries to store correct and total counts for each emotion category
correct_counts = {}
total_counts = {}

# Loop through each emotion sub-directory
for emotion_dir in os.listdir(base_directory):
    emotion_path = os.path.join(base_directory, emotion_dir)
    
    # Initialize counts for this emotion category
    correct_counts[emotion_dir.lower()] = 0
    total_counts[emotion_dir.lower()] = 0
    
    # Ensure it's a directory
    if os.path.isdir(emotion_path):
        # Loop through each image in the emotion sub-directory
        for filename in os.listdir(emotion_path):
            if filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".jpg"):
                img_path = os.path.join(emotion_path, filename)
                face_image = cv2.imread(img_path)
                face_image = preprocess_face(face_image)
                
                emotion_prediction = emotion_classifier.predict(face_image)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg].lower()

                # Compare the predicted emotion with the directory name (actual emotion)
                if emotion_text == emotion_dir.lower():
                    correct_counts[emotion_dir.lower()] += 1

                total_counts[emotion_dir.lower()] += 1

sum_of_percentages = 0.0
# Calculate and print the accuracy for each emotion category
for emotion, correct_count in correct_counts.items():
    total = total_counts[emotion]
    accuracy = (correct_count / total) * 100
    sum_of_percentages =+ accuracy
    print(f"Accuracy for {emotion.capitalize()}: {accuracy:.2f}%")
    
# print(f"Accuracy for all : {sum_of_percentages/7.0:.2f}%")

    
