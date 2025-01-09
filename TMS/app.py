from flask import Flask, Response
import cv2
import numpy as np
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timerse import ClockBasedTimer
import supervision as sv

app = Flask(__name__)

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)

source_video_path = "data/checkout4/video1.mp4"
zone_configuration_path = "data/checkout4/config.json"
weights = "3.pt"
device = "0"
confidence = 0.7
iou = 0.7
classes = []
log_file_path = "data/new.txt"
api_url = "http://localhost:8080/tms/events"
feed_device_id = 1


def video_stream_generator():
    model = YOLO(weights)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    frames_generator = sv.get_video_frames_generator(source_video_path)

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [ClockBasedTimer(api_url, feed_device_id, log_file_path) for _ in zones]

    for frame in frames_generator:
        results = model(frame, verbose=False, device=device, conf=confidence)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = detections.with_nms(threshold=iou)
        detections = tracker.update_with_detections(detections)

        annotated_frame = frame.copy()

        for idx, zone in enumerate(zones):
            # annotated_frame = sv.draw_polygon(
            #     scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            # )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)  # Update timer for the zone
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )

            # Optionally add labels showing time spent in the zone
            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )

        _, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_data = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(
        video_stream_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
