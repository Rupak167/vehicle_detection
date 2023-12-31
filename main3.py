from ultralytics import YOLO
import cv2
from tracker import Tracker
import pandas as pd

model = YOLO("best_v.pt")

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('road.mp4')

class_list = ['bus', 'car', 'motorbike', 'truck']

count = 0

tracker1 = Tracker()
tracker2 = Tracker()
tracker3 = Tracker()
tracker4 = Tracker()

counter1 = {}
counter2 = {}
counter3 = {}
counter4 = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list1 = []
    bus = []
    list2 = []
    car = []
    list3 = []
    motorbike = []
    list4 = []
    truck = []

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]

        if 'bus' in c:
            list1.append([x1, y1, x2, y2])
            bus.append(c)
        elif 'car' in c:
            list2.append([x1, y1, x2, y2])
            car.append(c)
        elif 'motorbike' in c:
            list3.append([x1, y1, x2, y2])
            motorbike.append(c)
        elif 'truck' in c:
            list4.append([x1, y1, x2, y2])
            truck.append(c)

    bbox1_idx = tracker1.update(list1)
    bbox2_idx = tracker2.update(list2)
    bbox3_idx = tracker3.update(list3)
    bbox4_idx = tracker4.update(list4)

    for bbox1 in bbox1_idx:
        x3, y3, x4, y4, id = bbox1
        if id not in counter1:
            counter1[id] = []
        counter1[id].extend(bus)

    for bbox2 in bbox2_idx:
        x2, y2, x3, y3, id1 = bbox2
        if id1 not in counter2:
            counter2[id1] = []
        counter2[id1].extend(car)

    for bbox3 in bbox3_idx:
        x0, y0, x1, y1, id0 = bbox3
        if id0 not in counter3:
            counter3[id0] = []
        counter3[id0].extend(motorbike)

    for bbox4 in bbox4_idx:
        x3, y3, x4, y4, id2 = bbox4
        if id2 not in counter4:
            counter4[id2] = []
        counter4[id2].extend(truck)

    for id, classes in counter1.items():
        if len(classes) > 0:
            most_predicted_class = max(set(classes), key=classes.count)
            cv2.putText(frame, f'Vehicle {id}: {most_predicted_class}', (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    for id, classes in counter2.items():
        if len(classes) > 0:
            most_predicted_class = max(set(classes), key=classes.count)
            cv2.putText(frame, f'Vehicle {id}: {most_predicted_class}', (12,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    for id, classes in counter3.items():
        if len(classes) > 0:
            most_predicted_class = max(set(classes), key=classes.count)
            cv2.putText(frame, f'Vehicle {id}: {most_predicted_class}', (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    for id, classes in counter4.items():
        if len(classes) > 0:
            most_predicted_class = max(set(classes), key=classes.count)
            cv2.putText(frame, f'Vehicle {id}: {most_predicted_class}', (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
