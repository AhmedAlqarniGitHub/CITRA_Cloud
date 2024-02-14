import cv2
import face_recognition
import os
import time
from google.cloud import storage

# Ensure a directory exists to store the cropped face images
if not os.path.exists('detected_faces'):
    os.makedirs('detected_faces')

# Open the video
cap = cv2.VideoCapture('http://192.168.100.154:5003/video_feed')

# Face counter
face_count = 0

start_time = time.time()
while True:
    # Read each frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Detect faces using face_recognition
    face_locations = face_recognition.face_locations(frame)

    # Loop over the detected faces
    for top, right, bottom, left in face_locations:
        # Increment the face count
        face_count += 1

        # Crop the face from the frame
        cropped_face = frame[top:bottom, left:right]

        # Save the cropped face
        filename = f"detected_faces/face_{face_count}.jpg"
        cv2.imwrite(filename, cropped_face)

        #copy to google cloud using json creds
        storage_client = storage.Client.from_service_account_json(
        'mx-project-404820-acb89a9e22e1.json')
        bucket = storage_client.get_bucket("kfupmmx")
        blob = bucket.blob(f"detected_faces/face_{face_count}.jpg")
        blob.upload_from_filename(filename)
        print("File uploaded to cloud storage file: ", filename)

end_time = time.time()

# Release the video capture object
cap.release()

cv2.destroyAllWindows()

print("total time = ",(end_time - start_time))
