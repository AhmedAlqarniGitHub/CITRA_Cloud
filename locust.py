from locust import HttpUser, task, between

# target the stress test to the deployed app on Cloud Run https://emotion-detection-app-bw5vqucpuq-ww.a.run.app
# The app is deployed on Cloud Run and the URL is https://emotion-detection-app-bw5vqucpuq-ww.a.run.app

# The stress test will be performed by sending a POST request to the deployed app on Cloud Run
# The POST request will contain the following data:
# - image: a file attachment
# - timestamp: a string
# - venue: a string
# - camera_id: a string
# The file attachment will be a valid image file
# The timestamp, venue, and camera_id will be valid strings

class MyTaskSet(HttpUser):
    @task
    def my_task(self):
        file_path = "./sad232.jpg"  # Make sure this file exists in the correct path
        with open(file_path, 'rb') as file:
            files = {'image': file}
            data = {"timestamp": "2/2/22", "venue": "locust", "camera_id": "111"}
            self.client.post("https://emotion-detection-app-bw5vqucpuq-ww.a.run.app/", files=files, data=data)

    host = "https://emotion-detection-app-bw5vqucpuq-ww.a.run.app/"
    wait_time = between(1, 5)
