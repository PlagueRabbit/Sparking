# app/yolo_nas.py (통합 버전)
import torch
import numpy as np
import cv2
from super_gradients.training import models
from app.utils.tracker import VehicleTracker
import supervision as sv


# ------------------- 기존 YOLO 클래스 백업 -------------------
# class YOLO:
#     def __init__(self, include_truck: bool = True, confidence_threshold: float = 0.5):
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {self.device}")
#         if torch.cuda.is_available():
#             print(torch.cuda.get_device_name(0))
#
#         self.model = models.get("yolo_nas_l", pretrained_weights="coco").to(self.device)
#         self.include_truck = include_truck
#         self.confidence_threshold = confidence_threshold
#
#     async def count_car(self, img, idx):
#         results = self.model.predict(img, conf=self.confidence_threshold, fuse_model=True)
#         results.save(output_path=f"predicted{idx}.jpg")
#
#         detections = sv.Detections.from_yolo_nas(results)
#         detections = detections[detections.confidence > self.confidence_threshold]
#
#         count_car, count_truck = 0, 0
#         for num in detections.class_id:
#             if num == 2:
#                 count_car += 1
#             elif num == 7 and self.include_truck:
#                 count_truck += 1
#
#         total = count_car + count_truck
#         print(f"Detected cars: {count_car}, trucks: {count_truck}, total: {total}")
#         return total


# ------------------- 통합 YOLO 클래스 -------------------
class YOLO:
    def __init__(self, include_truck: bool = True, confidence_threshold: float = 0.5):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))

        self.model = models.get("yolo_nas_l", pretrained_weights="coco").to(self.device)
        self.include_truck = include_truck
        self.confidence_threshold = confidence_threshold
        self.tracker = VehicleTracker()

    async def count_car(self, img, idx):
        results = self.model.predict(img, conf=self.confidence_threshold, fuse_model=True)
        results.save(output_path=f"predicted{idx}.jpg")

        detections = sv.Detections.from_yolo_nas(results)
        detections = detections[detections.confidence > self.confidence_threshold]

        count_car, count_truck = 0, 0
        for num in detections.class_id:
            if num == 2:
                count_car += 1
            elif num == 7 and self.include_truck:
                count_truck += 1

        total = count_car + count_truck
        print(f"Detected cars: {count_car}, trucks: {count_truck}, total: {total}")
        return total

    def detect_and_track(self, image_path: str, output_path: str) -> dict:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        preds = self.model.predict(image_rgb)
        bboxes = preds.prediction.bboxes_xyxy
        scores = preds.prediction.confidence
        labels = preds.prediction.labels

        car_indices = np.where(labels == 2)[0]
        detections = np.array([
            [*bboxes[i], scores[i]]
            for i in car_indices
        ]) if len(car_indices) > 0 else np.empty((0, 5))

        tracked_objects = self.tracker.update(detections)
        image_annotated = self.tracker.annotate_frame(image.copy(), tracked_objects)
        cv2.imwrite(output_path, image_annotated)

        return {
            "total_cars": len(tracked_objects),
            "long_parked": self.tracker.count_long_parked(),
        }


# 테스트 실행 예시
if __name__ == "__main__":
    yolo = YOLO()
    result = yolo.detect_and_track("app/test_img/sample_parking.jpg", "predicted1.jpg")
    print(result)