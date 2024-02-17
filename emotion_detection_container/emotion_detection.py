import base64
import json
import cv2
import numpy as np
from keras.models import load_model
from google.cloud import storage
from pymongo import MongoClient
import io

# MongoDB setup
mongo_client = MongoClient("mongodb+srv://engahmed:I0MH2jrfaDBQlZM1@finvis.sur9jkb.mongodb.net/?retryWrites=true&w=majority")
db = mongo_client.CITRA
emotions_collection = db.emotions

# Google Cloud Storage setup
bucket_name = "kfupmmx"
storage_client = storage.Client.from_service_account_json(
    'mx-project-404820-acb89a9e22e1.json')

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

def download_blob_to_memory(bucket_name, source_blob_name):
    """Downloads a blob from the bucket to memory."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    in_memory_file = io.BytesIO()
    blob.download_to_file(in_memory_file)
    in_memory_file.seek(0)
    return in_memory_file

def process_pubsub_message(event, context):
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    message = json.loads(pubsub_message)
    image_name = message['name']

    try:
        in_memory_file = download_blob_to_memory(bucket_name, image_name)
        data = np.frombuffer(in_memory_file.read(), dtype=np.uint8)
        face_image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Preprocess and predict
        face_image = preprocess_face(face_image)
        emotion_prediction = emotion_classifier.predict(face_image)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_dict[emotion_label_arg]

        # Save to MongoDB
        emotions_collection.insert_one({"image_name": image_name, "emotion": emotion_text})

        print(f'Processed file: {image_name} with detected emotion: {emotion_text}')
    except Exception as e:
        print(f'Error processing file {image_name}: {e}')

# The entry point for the Cloud Run service
def entry_point(request):
    envelope = request.get_json()
    if not envelope:
        msg = 'no Pub/Sub message received'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    if not isinstance(envelope, dict) or 'message' not in envelope:
        msg = 'invalid Pub/Sub message format'
        print(f'error: {msg}')
        return f'Bad Request: {msg}', 400

    pubsub_message = envelope['message']
    process_pubsub_message(pubsub_message, None)
    return ('', 204)