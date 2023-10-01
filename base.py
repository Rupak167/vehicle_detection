import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

model = YOLO('best.pt')

cap = cv2.VideoCapture('vid2.mp4')

class_names = ['bus', 'car', 'motorbike', 'truck', 'bicycle', 'cng', 'easy_bike', 'leguna', 'rickshaw', 'van']
class_counts = {'bus': 0, 'car': 0, 'motorbike': 0, 'truck': 0, 'bicycle': 0, 'cng': 0, 'easy_bike': 0, 'leguna': 0, 'rickshaw': 0, 'van': 0}


cv2.namedWindow('Vehicle Detection and Classification')
while True:
    ret, frame = cap.read()
    if ret:
        results = model.predict(frame)
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")
        for index, row in px.iterrows():

            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            class_id = int(row[5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_counts[class_names[class_id]] += 1

        cv2.putText(frame, 'Class counts:', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for class_name, count in class_counts.items():
            (text_width, text_height), _ = cv2.getTextSize(f'{class_name}: {count}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(frame, f'{class_name}: {count}', (10 + text_width, 20 + text_height * len(class_counts)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Vehicle Detection and Classification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()