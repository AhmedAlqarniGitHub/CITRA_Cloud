import psutil
from memory_profiler import memory_usage
import time
from function_emotion import detect_emotion


def report_memory_usage():
    memory_info = psutil.virtual_memory()
    used_memory_mb = memory_info.used / (1024 ** 2)  # Convert from bytes to MB
    return used_memory_mb

def main():
    # Path to the image you want to analyze
    mem_before = memory_usage(-1, interval=0.001, timeout=1)

    start_time = time.time()

    image_path = '../face_detection/detected_faces/face_2.jpg'

    # Call the emotion detection function
    detected_emotion = detect_emotion(image_path)

    end_time = time.time()

    # Measure memory after function call
    mem_after = memory_usage(-1, interval=0.001, timeout=1)
    print(f"Memory used: {max(mem_after) - min(mem_before)} MB")
    print(f"Time taken: {end_time - start_time} seconds")

    # Print the result
    print(f"Detected Emotion: {detected_emotion}")


if __name__ == "__main__":
    main()
