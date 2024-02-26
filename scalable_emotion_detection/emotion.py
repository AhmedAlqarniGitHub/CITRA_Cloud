import base64
import json
import cv2
import numpy as np
from keras.models import load_model
from pymongo import MongoClient
import io

# MongoDB setup
mongo_client = MongoClient("mongodb+srv://engahmed:I0MH2jrfaDBQlZM1@finvis.sur9jkb.mongodb.net/?retryWrites=true&w=majority")
db = mongo_client.CITRA
emotions_collection = db.emotions

# Emotion detection setup
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}
emotion_model_path = '/app/best_model.h5'
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

def preprocess_face(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, emotion_target_size)
    face_image = face_image.astype('float32') / 255.0
    face_image = np.expand_dims(face_image, 0)
    face_image = np.expand_dims(face_image, -1)
    return face_image

def process_image(image_data):
    try:
        data = np.frombuffer(image_data, dtype=np.uint8)
        face_image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Preprocess and predict
        face_image = preprocess_face(face_image)
        emotion_prediction = emotion_classifier.predict(face_image)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_dict[emotion_label_arg]

        # Save to MongoDB
        emotions_collection.insert_one({"emotion": emotion_text})

        print(f'Processed image with detected emotion: {emotion_text}')
    except Exception as e:
        print(f'Error processing image: {e}')

# The entry point for the Cloud Run service
def entry_point(request):
    if request.method != 'POST':
        return 'Only POST requests are accepted', 405

    image_file = request.files.get('image')
    if not image_file:
        return 'No image file found in the request', 400

    image_data = image_file.read()
    process_image(image_data)
    return ('', 204)

if __name__ == "__main__":
    from flask import Flask, request
    app = Flask(__name__)

    @app.route('/', methods=['POST'])
    def index():
        return entry_point(request)

    app.run(host='0.0.0.0', port=8080, debug=True)
