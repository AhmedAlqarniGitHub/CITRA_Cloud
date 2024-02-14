import cv2
from flask import Flask, Response

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # Use 0 for default camera

def generate():
    """Video streaming generator function."""
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, jpeg = cv2.imencode('.jpg', gray_frame)
        if not ret:
            continue
        frame_byte = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_byte + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, threaded=True)
