from ultralytics import YOLO
import cv2
import cvzone
import torch
from sort import *
# import math


device = "cuda" if torch.cuda.is_available() else "cpu"

cap = cv2.VideoCapture('../chapter5/images/cars.mp4')

model = YOLO("../yolo-weight/yolov8n.pt").to(device)
classNames = model.names #classNames(dictionary) = {0:'person',1:'bicycle,2:'car',...}

blank = np.zeros((720,1280), np.uint8) ## numpy --> (h,w,channels)
mask = cv2.rectangle(blank,(245,7),(1162,312),(255,255,255),-1)

## Tracking(make an instance)
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

# limits = [183,154,1068,145]
limits = [149, 181, 1095, 181]
count = []



while True:
    isTrue, img = cap.read()
    img = cv2.resize(img,(1280,720)) ## opencv --> (w,h)
    imgMs = cv2.bitwise_and(img, img, mask = mask)

    graphics = cv2.imread('graphics.png',cv2.IMREAD_UNCHANGED)
    graphics = cv2.resize(graphics,(195,135))
    graphics = cv2.cvtColor(graphics, cv2.COLOR_BGR2BGRA)
    img = cvzone.overlayPNG(img,graphics,(0,0))

    result = model(imgMs, stream = True, device = device)#مدل روی فریم اعمال میشه و خروچیش مثل بسته کامل اطلاعات شامل اشیا پیدا شده(جعبه)/  stream= true-->خروجی رو فریم به فریم تحویل میگیری

    detections = np.empty((0, 5))


    for r in result: # روی نتیجه خام تشخیص مدل حرکت کن/ (محتویات جعبه)تمام اطلاعات اشیا پیدا شده = r
        boxes = r.boxes #همه اشیایی که مدل تو این فریم پیدا کرده
        for box in boxes:#روی هر شی تشخیص داده شده جرکت میکنه، مثلا بار اول ادم، بار دوم ماشین
            x1, y1, x2, y2 = box.xyxy[0]   #مختصات شی خاص
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = x2-x1, y2-y1
            # conf = math.ceil((box.conf[0]*100))/100   --> گرد کردن استاندارده round گرد کردن رو بالاست ولی ceil
            conf = round(float(box.conf[0]),2)
            cls = int(box.cls[0]) #شماره کلاس شیء
            currentClass = classNames[cls]
            if  currentClass == "car" and conf > 0.3: #or currentClass == "bicycle"
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0,x1),max(30,y1)),scale=1,thickness=2,offset=8)#offset--> اندازه مستطیل بنفش #classNames[cls] --> هست cls اسم کلاسی که شمارش برابر با

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections) #format = numpy array(5 values)
    cv2.line(img, (limits[0],limits[1]), (limits[2],limits[3]), (0,0,255), 4)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(30, y1)), scale=1, thickness=2, offset=8)

        cx,cy = x1+w//2, y1+h//2

        if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[3]+15:
            if count.count(id) == 0: # count the number of time that this(id) is present in this list[count]
                count.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0),10)

    cv2.putText(img,str(len(count)), (118,91), cv2.FONT_HERSHEY_PLAIN, 3, (37, 232, 125), 3)


    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
