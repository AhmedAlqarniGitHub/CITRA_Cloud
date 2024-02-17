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
- Create a single container that can do Cloud Functionality completely (In-Progress)
- Create Container AutoScale in Google Cloud (Not-Started)
- Map Load Balancer to Created AutoScale instances (Not-Started)
