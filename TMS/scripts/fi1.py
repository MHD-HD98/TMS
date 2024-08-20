import argparse
import cv2
from flask import Flask, Response, render_template_string
import time

# Set up Flask app
app = Flask(__name__)

video_capture = None
frame_rate = 80  # Set FPS to 29

@app.route('/')
def index():
    # Video streaming home page
    return render_template_string('<html><body><h1>Video Stream</h1><img src="/video_feed"></body></html>')

def gen_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in the stream format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Delay to maintain real-time playback at 29 FPS
            time.sleep(1 / frame_rate)

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main(video_file: str) -> None:
    global video_capture

    video_capture = cv2.VideoCapture(video_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to stream a video using HTTP protocol with Flask."
    )
    parser.add_argument(
        "--video_file",
        type=str,
        required=True,
        help="Path to the .mp4 video file to stream.",
    )
    args = parser.parse_args()

    main(video_file=args.video_file)

    # Start the Flask server
    app.run(host="localhost", port=5000, threaded=True)
