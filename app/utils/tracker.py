# app/utils/tracker.py
from datetime import datetime, timedelta
import numpy as np
from app.utils.sort.sort import Sort


#장기주차 기준
LONG_PARKING_THRESHOLD = timedelta(seconds=5)

class VehicleTracker:
    def __init__(self):
        self.tracker = Sort()
        self.vehicle_data = {}  # vehicle_id: {"start_time": datetime, "last_seen": datetime, "bbox": [x1, y1, x2, y2]}

    def update(self, detections):
        """
        :param detections: np.array([[x1, y1, x2, y2, conf], ...])
        :return: tracked_objects: list of dicts
        """
        tracked_objects = []
        results = self.tracker.update(detections)
        now = datetime.now()

        for result in results:
            x1, y1, x2, y2, vehicle_id = result.astype(int)
            bbox = [x1, y1, x2, y2]

            if vehicle_id not in self.vehicle_data:
                self.vehicle_data[vehicle_id] = {
                    "start_time": now,
                    "last_seen": now,
                    "bbox": bbox
                }
            else:
                self.vehicle_data[vehicle_id]["last_seen"] = now
                self.vehicle_data[vehicle_id]["bbox"] = bbox

            duration = now - self.vehicle_data[vehicle_id]["start_time"]
            is_long_parked = duration > LONG_PARKING_THRESHOLD

            tracked_objects.append({
                "id": vehicle_id,
                "bbox": bbox,
                "duration": duration,
                "is_long_parked": is_long_parked
            })

        self._remove_missing_vehicles(now)
        return tracked_objects

    def _remove_missing_vehicles(self, current_time, max_disappear_time=timedelta(minutes=10)):
        to_delete = []
        for vid, data in self.vehicle_data.items():
            if current_time - data["last_seen"] > max_disappear_time:
                to_delete.append(vid)

        for vid in to_delete:
            del self.vehicle_data[vid]

    def count_long_parked(self):
        now = datetime.now()
        return sum(
            1 for data in self.vehicle_data.values()
            if now - data["start_time"] > LONG_PARKING_THRESHOLD
        )

    def annotate_frame(self, frame, tracked_objects):
        import cv2
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            color = (0, 255, 0) if obj["is_long_parked"] else (255, 0, 0)
            label = f"ID {obj['id']} {'LONG' if obj['is_long_parked'] else ''}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
