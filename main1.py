from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from tracker import*
import cvzone


model = YOLO("best.pt")


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        point = [x, y]
        print(point)



cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('road.mp4')

# my_file = open("coco.txt", "r")
# data = my_file.read()
# class_list = data.split("\n")
class_list = ['bus', 'car', 'motorbike', 'truck', 'bicycle', 'cng', 'easy_bike', 'leguna', 'rickshaw', 'van']
#print(class_list)

count=0
cy1=328

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
            x3,y3,x4,y4,id=bbox1
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i} num {len(counter1)}',(x3,y3),0.5,1)
               if counter1.count(id)==0:
                  counter1.append(id)

    for bbox2 in bbox2_idx:
        for i in car:
            x2,y2,x3,y3,id1=bbox2
            cxc=int(x2+x3)//2
            cyc=int(y2+y3)//2
            if cyc<(cy1+offset) and cyc>(cy1-offset):
               cv2.circle(frame,(cxc,cyc),4,(0,255,0),-1)
               cv2.rectangle(frame,(x2,y2),(x3,y3),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i} num {len(counter2)}',(x2,y2),0.5,1)
               if counter2.count(id1)==0:
                  counter2.append(id1)

    for bbox3 in bbox3_idx:
        for i in motorbike:
            x0,y0,x1,y1,id0=bbox3
            cxm=int(x0+x1)//2
            cym=int(y0+y1)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x0,y0),(x1,y1),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i} num {len(counter3)}',(x0,y0),0.5,1)
               if counter3.count(id0)==0:
                  counter3.append(id0)

    

    for bbox4 in bbox4_idx:
        for i in truck:
            x3,y3,x4,y4,id2=bbox4
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i} num {len(counter4)}',(x3,y3),0.5,1)
               if counter4.count(id2)==0:
                  counter4.append(id2)


    

    for bbox5 in bbox5_idx:
        for i in bicycle:
            x3,y3,x4,y4,id3=bbox5
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i} num {len(counter5)}',(x3,y3),0.5,1)
               if counter5.count(id3)==0:
                  counter5.append(id3)



    for bbox6 in bbox6_idx:
        for i in cng:
            x3,y3,x4,y4,id4=bbox6
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i} num {len(counter6)}',(x3,y3),0.5,1)
               if counter6.count(id4)==0:
                  counter6.append(id4)

    for bbox7 in bbox7_idx:
        for i in cng:
            x3,y3,x4,y4,id4=bbox7
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i} num {len(counter7)}',(x3,y3),0.5,1)
               if counter7.count(id4)==0:
                  counter7.append(id4)



    for bbox8 in bbox8_idx:
        for i in cng:
            x3,y3,x4,y4,id4=bbox8
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i} num {len(counter8)}',(x3,y3),0.5,1)
               if counter8.count(id4)==0:
                  counter8.append(id4)



    for bbox9 in bbox9_idx:
        for i in cng:
            x3,y3,x4,y4,id4=bbox9
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i} num {len(counter9)}',(x3,y3),0.5,1)
               if counter9.count(id4)==0:
                  counter9.append(id4)

    for bbox10 in bbox10_idx:
        for i in cng:
            x3,y3,x4,y4,id4=bbox10
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i} num {len(counter10)}',(x3,y3),0.5,1)
               if counter10.count(id4)==0:
                  counter10.append(id4)
    
    cv2.line(frame,(2,cy1),(1050,cy1),(255,255,255),1)

    bus=(len(counter1))
    car = (len(counter2))
    motorbike = (len(counter3))
    truck = (len(counter4))
    bicycle = (len(counter5))
    cng = (len(counter6))
    easy_bike = (len(counter7))
    leguna = (len(counter8))
    rickshaw = (len(counter9))
    van = (len(counter10))
    
    cvzone.putTextRect(frame,f'bus:-{bus}',(12,20),1,1)
    cvzone.putTextRect(frame,f'car:-{car}',(12,45),1,1)
    cvzone.putTextRect(frame,f'motorbike:-{motorbike}',(12,70),1,1)
    cvzone.putTextRect(frame,f'truck:-{truck}',(12,95),1,1)
    cvzone.putTextRect(frame,f'bicycle:-{bicycle}',(12,120),1,1)

    cvzone.putTextRect(frame,f'cng:-{cng}',(140,20),1,1)
    cvzone.putTextRect(frame,f'easy_bike:-{easy_bike}',(140,45),1,1)
    cvzone.putTextRect(frame,f'leguna:-{leguna}',(140,70),1,1)
    cvzone.putTextRect(frame,f'rickshaw:-{rickshaw}',(140,95),1,1)
    cvzone.putTextRect(frame,f'van:-{van}',(140,120),1,1)


    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()