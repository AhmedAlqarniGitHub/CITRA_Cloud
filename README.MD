first do the following

1-  install cmake

sudo apt install cmake

brew install cmake (mac)

2- install python pip

3- install requirements.txt

pip install -r requirements.txt

-------
Load Balancer:
- Mapped to Created Instances of Containers

-------
Cloud Functionality:
- Message Received:
    - POST: Image file
    - POST Parameters (JSON): Camera ID; Event ID; Timestamp
- Cloud Processing:
    - Extract Image file
    - Extract Camera ID; Event ID; Timestamp from body (JSON)
    - Process Emotion Detection on Image
    - Write to MongoDB Emotion Detected; Camera ID; Event ID; Timestamp
    - Delete Image file
- Repeat.


===== Work Log:
- Create a single program that can do Cloud Functionality completely (COMPLETED)
- Create a single Container that can do Cloud Functionality and Write to MongoDB (COMPLETED)
- Create Container AutoScale in Google Cloud (In-Progress)
- Map Load Balancer to Created AutoScale instances (Not-Started)

===== Code Run:
docker build -t emotion-detection-app .  --platform linux/amd64
docker run -p 8088:8080 emotion-detection-app

docker tag emotion-detection-app me-central1-docker.pkg.dev/mx-project-citra/citra2/emotion-detection-app:feb272
docker push me-central1-docker.pkg.dev/mx-project-citra/citra2/emotion-detection-app:feb272

--platform linux/amd64