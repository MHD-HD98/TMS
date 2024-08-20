from datetime import datetime
from typing import Dict

import numpy as np
import supervision as sv
import requests

# Define start and end events for specific tracker IDs
EVENT_START = {1: "CGO_AFT_OT",2: "PBB_FWD_ON"}
EVENT_END = {1: "CGO_AFT_CT", 2: "PBB_FWD_OFF"}




class FPSBasedTimer:
    """
    A timer that calculates the duration each object has been detected based on frames
    per second (FPS).

    Attributes:
        fps (int): The frame rate of the video stream, used to calculate time durations.
        frame_id (int): The current frame number in the sequence.
        tracker_id2frame_id (Dict[int, int]): Maps each tracker's ID to the frame number
            at which it was first detected.
    """

    def __init__(self, fps: int = 30) -> None:
        """Initializes the FPSBasedTimer with the specified frames per second rate.

        Args:
            fps (int, optional): The frame rate of the video stream. Defaults to 30.
        """
        self.fps = fps
        self.frame_id = 0
        self.tracker_id2frame_id: Dict[int, int] = {}

    def tick(self, detections: sv.Detections) -> np.ndarray:
        """Processes the current frame, updating time durations for each tracker.

        Args:
            detections: The detections for the current frame, including tracker IDs.

        Returns:
            np.ndarray: Time durations (in seconds) for each detected tracker, since
                their first detection.
        """
        self.frame_id += 1
        times = []

        for tracker_id in detections.tracker_id:
            self.tracker_id2frame_id.setdefault(tracker_id, self.frame_id)

            start_frame_id = self.tracker_id2frame_id[tracker_id]
            time_duration = (self.frame_id - start_frame_id) / self.fps
            times.append(time_duration)

        return np.array(times)


class ClockBasedTimer:
    """
    A timer that calculates the duration each object has been detected based on the
    system clock.

    Attributes:
        tracker_id2start_time (Dict[int, datetime]): Maps each tracker's ID to the
            datetime when it was first detected.
        tracker_id2end_time (Dict[int, datetime]): Maps each tracker's ID to the
            datetime when it was last detected.
    """

    def __init__(self, api_url: str, feed_device_id: int, log_file_path: str) -> None:
        """Initializes the ClockBasedTimer with a file to log times."""
        self.tracker_id2start_time: Dict[int, datetime] = {}
        self.tracker_id2end_time: Dict[int, datetime] = {}
        self.log_file_path = log_file_path
        self.api_url = api_url
        self.feed_device_id = feed_device_id

    def log_detection_times(self, tracker_id: int, start_time: datetime, end_time: datetime, event_code: str) -> None:
        """Logs the detection times to a text file."""
        duration = (end_time - start_time).total_seconds()
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(
                f"Tracker ID: {tracker_id}\nEvent Code: {event_code}\nStart Time: {start_time}\nEnd Time: {end_time}\nDuration: {duration:.2f} seconds\n\n"
            )

    def push_event(self, event_time: datetime, event_code: str) -> None:
        """Pushes an event to the API.

        Args:
            event_time (datetime): The time of the event.
            event_code (str): The event code.
        """
        event_data = {
            "eventInstance": event_time.strftime('%Y-%m-%dT%H:%M:%S'),
            "feedDeviceId": self.feed_device_id,
            "eventCode": event_code
        }
        response = requests.post(self.api_url, json=event_data)
        if response.status_code == 200:
            print(f"Successfully pushed event: {event_data}")
        else:
            print(f"Failed to push event: {event_data} - Status Code: {response.status_code}")

    def tick(self, detections: sv.Detections) -> np.ndarray:
        """Processes the current frame, updating time durations for each tracker.

        Args:
            detections: The detections for the current frame, including tracker IDs.

        Returns:
            np.ndarray: Time durations (in seconds) for each detected tracker, since
                their first detection.
        """
        current_time = datetime.now()
        times = []

        detected_tracker_ids = set(detections.tracker_id)

        # Update start and end times for detected trackers
        for tracker_id in detected_tracker_ids:
            if tracker_id not in self.tracker_id2start_time:
                self.tracker_id2start_time[tracker_id] = current_time
                event_code = EVENT_START.get(tracker_id, )
                self.push_event(current_time, event_code)
                print(f"Start time for tracker {tracker_id}: {current_time.strftime('%Y-%m-%dT%H:%M:%S')}, Event Code: {event_code}")
            self.tracker_id2end_time[tracker_id] = current_time

        # Handle trackers that are no longer detected
        for tracker_id in list(self.tracker_id2start_time.keys()):
            if tracker_id not in detected_tracker_ids:
                end_time = self.tracker_id2end_time.pop(tracker_id, current_time)
                start_time = self.tracker_id2start_time.pop(tracker_id)
                event_code = EVENT_END.get(tracker_id,)
                self.push_event(end_time, event_code)
                print(f"End time for tracker {tracker_id}: {end_time.strftime('%Y-%m-%dT%H:%M:%S')}, Event Code: {event_code}")
                self.log_detection_times(tracker_id, start_time, end_time, event_code)

        # Calculate time durations for currently detected trackers
        for tracker_id in detected_tracker_ids:
            start_time = self.tracker_id2start_time[tracker_id]
            time_duration = (current_time - start_time).total_seconds()
            times.append(time_duration)

        return np.array(times)
