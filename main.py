import cv2 as cv
from ultralytics import YOLO
import numpy as np
from sort import *
from utils import get_car, read_plate, write_csv

results = {}

tracker = Sort()

car_model = YOLO("yolo11n.pt")
license_model = YOLO("./plate_detector/runs/detect/train3/weights/best.pt")

capture = cv.VideoCapture('./datasets/test_video.mp4')

vehicles = [2, 3, 5, 6, 7]  # cars, motorcycle, bus, train, truck

frame_count = -1
ret = True

while ret:
    frame_count += 1
    ret, frame = capture.read()

    if ret:
        results[frame_count] = {}
        detections = car_model(frame)[0]
        detections_boxes = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_boxes.append([x1, y1, x2, y2, score])

        track_ids = tracker.update(np.asarray(detections_boxes))

        plates = license_model(frame)[0]
        for plate in plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = plate

            xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, track_ids)
            
            if car_id != -1:
                cropped = frame[int(y1):int(y2), int(x1):int(x2), :]

                cropped_gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
                _, cropped_threshold = cv.threshold(
                    cropped_gray, 64, 255, cv.THRESH_BINARY_INV)

                text, text_score = read_plate(cropped_threshold)

                if text is not None:
                    results[frame_count][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': text,
                                                                    'bbox_score': score,
                                                                    'text_score': text_score}}

write_csv(results, './datasets/results.csv')