import argparse
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn

# Set up FastAPI app
app = FastAPI()

templates = Jinja2Templates(directory="templates")

video_capture = None

@app.get("/")
async def index(request: Request):
    # Video streaming home page
    return templates.TemplateResponse("index.html", {"request": request})

def gen_frames():
    global video_capture
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

@app.get("/video_feed")
async def video_feed():
    # Video streaming route
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

def main(video_file: str) -> None:
    global video_capture
    video_capture = cv2.VideoCapture(video_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to stream a video using HTTP protocol with FastAPI."
    )
    parser.add_argument(
        "--video_file",
        type=str,
        required=True,
        help="Path to the .mp4 video file to stream.",
    )
    args = parser.parse_args()

    main(video_file=args.video_file)

    # Start the FastAPI server
    uvicorn.run(app, host="localhost", port=8000)
