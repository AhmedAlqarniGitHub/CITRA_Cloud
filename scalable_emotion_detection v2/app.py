import cv2
import numpy as np
from keras.models import load_model
from pymongo import MongoClient
import os
from bson import ObjectId
from datetime import datetime
from bson.objectid import ObjectId

# MongoDB setup
mongo_client = MongoClient("mongodb+srv://ahmedalg4321:citra321@cluster0.u2aiu58.mongodb.net/?retryWrites=true&w=majority")
db = mongo_client.citra
emotions_collection = db.events

# Emotion detection setup
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}
#emotion_model_path = '/app/best_model.h5'
emotion_model_path = './best_model.h5'
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

def preprocess_face(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, emotion_target_size)
    face_image = face_image.astype('float32') / 255.0
    face_image = np.expand_dims(face_image, 0)
    face_image = np.expand_dims(face_image, -1)
    return face_image

def process_image(image_data, detectionTime, eventId, cameraId):
    try:
        data = np.frombuffer(image_data, dtype=np.uint8)
        face_image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Preprocess and predict
        face_image = preprocess_face(face_image)
        emotion_prediction = emotion_classifier.predict(face_image)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_dict[emotion_label_arg]

        # Convert detectionTime from string to datetime object
        if isinstance(detectionTime, str):
            detectionTime = datetime.fromisoformat(detectionTime)

        # Save to MongoDB
        # Emotion data to be added
        emotion_data = {
            "_id" : ObjectId(),
            "camera": ObjectId(cameraId),
            "detectionTime": detectionTime,
            "detectedEmotion": emotion_text
        }
        event_id = ObjectId(eventId)
        # Push the new emotion data into the 'emotions' array of the event with the matching eventId
        result = emotions_collection.update_one(
            {"_id": event_id},  # Match the event document by eventId
            {"$push": {"emotions": emotion_data}}  # Push the new emotion object into the emotions array
        )

        if result.matched_count > 0:
            if result.modified_count > 0:
                print(f'Successfully updated document with eventId: {eventId}')
            else:
                    print(f'Document with eventId: {eventId} was found, but no changes were made.')
        else:
            print(f'No document found with eventId: {eventId}')
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

    detectionTime = request.form.get('detectionTime')
    eventId = request.form.get('eventId')
    cameraId = request.form.get('cameraId')
    if not (detectionTime and eventId and cameraId):
        return 'Missing detectionTime, eventId, or cameraId in the request', 400

    image_data = image_file.read()
    process_image(image_data, detectionTime, eventId, cameraId)
    return ('', 204)


if __name__ == "__main__":
    from flask import Flask, request
    app = Flask(__name__)
    print("Server running on port", os.environ["PORT"])
    @app.route('/', methods=['POST'])
    def index():
        return entry_point(request)

    app.run(host='0.0.0.0', port= os.environ["PORT"], debug=True)
    #app.run(host='0.0.0.0', port= 8088, debug=True)
    print("Server running on port")
