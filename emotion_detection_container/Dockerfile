# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8070 available to the world outside this container
EXPOSE 8070

# Define environment variable
ENV MODEL_PATH /app/best_model.h5

# Run app.py when the container launches
CMD ["python", "app.py"]
