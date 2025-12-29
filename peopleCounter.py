### People-counting-on-escalator

from ultralytics import YOLO
import cv2
import cvzone
import torch
import numpy as np
from sort import Sort


## Paths
VIDEO_PATH = "assets/people3.mp4"
MODEL_PATH = "../yolo-weight/yolov8n.pt"
GRAPHICS_PATH = "assets/es_graphics.png"


device = "cuda" if torch.cuda.is_available() else "cpu"

cap = cv2.VideoCapture(VIDEO_PATH)

model = YOLO(MODEL_PATH).to(device)
classNames = model.names

## Create ROI mask (escalators area)
blank = np.zeros((640,720), np.uint8)
mask = cv2.rectangle(blank,(10,100),(330,460),(255,255,255),-1)

## Tracking
tracker = Sort(max_age=30,min_hits=3,iou_threshold=0.3)

## Define counting lines (Up / Down directions)
limitsUp = [40,300,150,300]
limitsDown = [190,250,300,250]

## Store unique IDs to avoid double counting
countUp = set()
countDown = set()

## Load graphics once
graphics = cv2.imread(GRAPHICS_PATH, cv2.IMREAD_UNCHANGED)
graphics = cv2.resize(graphics, (200, 80))
graphics = cv2.cvtColor(graphics, cv2.COLOR_BGR2BGRA)

while True:
    isTrue, img = cap.read()
    if not isTrue:
        break
    img = cv2.resize(img,(720, 640))
    imgMs = cv2.bitwise_and(img, img, mask = mask)

    ## Overlay Graphics
    img = cvzone.overlayPNG(img,graphics,(5,15))

    result = model(imgMs, stream = True, device = device)
    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = x2-x1, y2-y1
            conf = round(float(box.conf[0]),2)
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if  currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsUp[0],limitsUp[1]), (limitsUp[2],limitsUp[3]), (0,0,255), 4)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 4)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(30, y1)), scale=1, thickness=2, offset=8)

        cx,cy = x1+w//2, y1+h//2

        if limitsUp[0]<cx<limitsUp[2] and limitsUp[1]-15<cy<limitsUp[3]+15:
            if id not in countUp:
                countUp.add(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0),10)

        if limitsDown[0]<cx<limitsDown[2] and limitsDown[1]-15<cy<limitsDown[3]+15:
            if id not in countDown:
                countDown.add(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0),10)

    ## Draw limit lines
    cv2.putText(img,str(len(countUp)), (55,85), cv2.FONT_HERSHEY_PLAIN, 3, (37, 232, 125), 2)
    cv2.putText(img,str(len(countDown)), (129,85), cv2.FONT_HERSHEY_PLAIN, 3, (22, 3, 166), 2)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()