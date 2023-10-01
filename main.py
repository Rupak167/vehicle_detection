from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from tracker import*
import cvzone
model = YOLO("yolov8l.pt")
def RGB(event, x, y, flags, param):
  if event == cv2.EVENT_MOUSEMOVE :
    point = [x, y]
    print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('vid2.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
cy1=424

tracker1=Tracker()
tracker2=Tracker()
tracker3=Tracker()
tracker4=Tracker()
tracker5=Tracker()
tracker6=Tracker()
tracker7=Tracker()
tracker8=Tracker()
tracker9=Tracker()
tracker10=Tracker()


counter1=[]
counter2=[]
counter3=[]
counter4=[]
counter5=[]
counter6=[]
counter7=[]
counter8=[]
counter9=[]
counter10=[]
offset=6
while True:  
  ret,frame = cap.read()
  if not ret:
      break
  count += 1
  if count % 3 != 0:
      continue
  
  frame=cv2.resize(frame,(1020,500))
  results=model.predict(frame)
  # print(results)
  a=results[0].boxes.data
  px=pd.DataFrame(a).astype("float")
  # print(px)
  list1=[]
  bus=[]
  list2=[]
  car=[]
  list3=[]
  motorbike=[]
  list4=[]
  truck=[]
  list5=[]
  bicycle=[]
  list6=[]
  cng=[]
  list7=[]
  easy_bike=[]
  list8=[]
  leguna=[]
  list9=[]
  rickshaw=[]
  list10=[]
  van=[]
  for index,row in px.iterrows():
    # print(row)
    x1=int(row[0])
    y1=int(row[1])
    x2=int(row[2])
    y2=int(row[3])
    d=int(row[5])
    c=class_list[d]
    # names = ['bus', 'car', 'motorbike', 'truck', 'bicycle', 'cng', 'easy_bike', 'leguna', 'rickshaw', 'van']
    if 'bus' in c:
      list1.append([x1,y1,x2,y2])
      bus.append(c)
    elif 'car' in c:
      list2.append([x1,y1,x2,y2])
      car.append(c)
    elif 'motorbike' in c:
      list3.append([x1,y1,x2,y2])
      motorbike.append(c)
    elif 'truck' in c:
      list4.append([x1,y1,x2,y2])
      truck.append(c)
    elif 'bicycle' in c:
      list5.append([x1,y1,x2,y2])
      bicycle.append(c)
    elif 'cng' in c:
      list6.append([x1,y1,x2,y2])
      cng.append(c)
    elif 'easy_bike' in c:
      list7.append([x1,y1,x2,y2])
      easy_bike.append(c)
    elif 'leguna' in c:
      list8.append([x1,y1,x2,y2])
      leguna.append(c)
    elif 'rickshaw' in c:
      list9.append([x1,y1,x2,y2])
      rickshaw.append(c)
    elif 'van' in c:
      list10.append([x1,y1,x2,y2])
      van.append(c)
          
  bbox1_idx=tracker1.update(list1)
  bbox2_idx=tracker2.update(list2)
  bbox3_idx=tracker3.update(list3)
  bbox4_idx=tracker4.update(list4)
  bbox5_idx=tracker5.update(list5)
  bbox6_idx=tracker6.update(list6)
  bbox7_idx=tracker7.update(list7)
  bbox8_idx=tracker8.update(list8)
  bbox9_idx=tracker9.update(list9)
  bbox10_idx=tracker10.update(list10)

  for bbox1 in bbox1_idx:
      for i in bus:
          x3,y3,x4,y4,id1=bbox1
          cxb=int(x3+x4)//2
          cyb=int(y3+y4)//2
          if cyb<(cy1+offset) and cyb>(cy1-offset):
              cv2.circle(frame,(cxb,cyb),4,(0,255,0),-1)
              cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
              cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
              if counter1.count(id1)==0:
                counter1.append(id1)
  for bbox2 in bbox2_idx:
      for i in car:
          x5,y5,x6,y6,id2=bbox2
          cxc=int(x5+x6)//2
          cyc=int(y5+y6)//2
          if cyc<(cy1+offset) and cyc>(cy1-offset):
              cv2.circle(frame,(cxc,cyc),4,(0,255,0),-1)
              cv2.rectangle(frame,(x5,y5),(x6,y6),(0,0,255),1)
              cvzone.putTextRect(frame,f'{id2}',(x5,y5),1,1)
              if counter2.count(id2)==0:
                counter2.append(id2)
  
  cv2.line(frame,(2,cy1),(794,cy1),(0,0,255),2)

  bus=(len(counter1))
  car = (len(counter2))
  cvzone.putTextRect(frame,f'bus:-{bus}',(19,30),2,1)
  cvzone.putTextRect(frame,f'car:-{car}',(18,71),2,1)


  cv2.imshow("RGB", frame)
  if cv2.waitKey(0)&0xFF==27:
     break

cap.release()
cv2.destroyAllWindows()