import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone

model=YOLO('yolov8l.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('road.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
cy1=320


tracker1=Tracker()
tracker2=Tracker()
tracker3=Tracker()
tracker4=Tracker()
tracker5=Tracker()



counter1=[]
counter2=[]
counter3=[]
counter4=[]
counter5=[]


offset=6
while True:    
    ret,frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 4 != 0:
        continue
    
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list1=[]
    motorcycle=[]
    list2=[]
    car=[]
    list3=[]
    truck=[]
    list4=[]
    bicycle=[]
    list5=[]
    bus=[]

    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]

        if 'bus' in c:
            list1.append([x1,y1,x2,y2])
            bus.append(c)
        elif 'car' in c:
            list2.append([x1,y1,x2,y2])
            car.append(c)
        elif 'truck' in c:
            list4.append([x1,y1,x2,y2])
            truck.append(c)
        elif 'bicycle' in c:
            list5.append([x1,y1,x2,y2])
            bicycle.append(c)
        elif 'motorcycle' in c:
            list1.append([x1,y1,x2,y2])
            motorcycle.append(c)
            
    bbox1_idx=tracker1.update(list1)
    bbox2_idx=tracker2.update(list2)
    bbox3_idx=tracker3.update(list3)
    bbox4_idx=tracker4.update(list4)
    bbox5_idx=tracker5.update(list5)


    for bbox1 in bbox1_idx:
        for i in motorcycle:
            x0,y0,x1,y1,id0=bbox1
            cxm=int(x0+x1)//2
            cym=int(y0+y1)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x0,y0),(x1,y1),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i}',(x0,y0),0.5,1)
               if counter1.count(id0)==0:
                  counter1.append(id0)

    for bbox2 in bbox2_idx:
        for i in car:
            x2,y2,x3,y3,id1=bbox2
            cxc=int(x2+x3)//2
            cyc=int(y2+y3)//2
            if cyc<(cy1+offset) and cyc>(cy1-offset):
               cv2.circle(frame,(cxc,cyc),4,(0,255,0),-1)
               cv2.rectangle(frame,(x2,y2),(x3,y3),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i}',(x2,y2),0.5,1)
               if counter2.count(id1)==0:
                  counter2.append(id1)

    for bbox3 in bbox3_idx:
        for i in truck:
            x3,y3,x4,y4,id2=bbox3
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i}',(x3,y3),0.5,1)
               if counter3.count(id2)==0:
                  counter3.append(id2)


    

    for bbox4 in bbox4_idx:
        for i in bicycle:
            x3,y3,x4,y4,id3=bbox4
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i}',(x3,y3),0.5,1)
               if counter4.count(id3)==0:
                  counter4.append(id3)



    for bbox5 in bbox5_idx:
        for i in bus:
            x3,y3,x4,y4,id4=bbox5
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            if cym<(cy1+offset) and cym>(cy1-offset):
               cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
               cvzone.putTextRect(frame,f'{i}',(x3,y3),0.5,1)
               if counter5.count(id4)==0:
                  counter5.append(id4)
   

    cv2.line(frame,(2,cy1),(994,cy1),(0,0,255),2)

  
    motorcyclec=(len(counter1))
    car=(len(counter2))
    truck=(len(counter3))
    bicycle=(len(counter4))
    bus=(len(counter5))
    cvzone.putTextRect(frame,f'motorcyclec:-{motorcyclec}',(12,20),1,1)
    cvzone.putTextRect(frame,f'car:-{car}',(12,45),1,1)
    cvzone.putTextRect(frame,f'truck:-{truck}',(12,70),1,1)
    cvzone.putTextRect(frame,f'bicycle:-{bicycle}',(12,95),1,1)
    cvzone.putTextRect(frame,f'bus:-{bus}',(12,120),1,1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if cv2.waitKey(1)&0xFF==27:
    #     break

cap.release()
cv2.destroyAllWindows()