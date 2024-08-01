from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("best_11.pt")

# object classes
classNames = ["Matang", "Agak Matang", "Mentah"]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            center_x = (x1 + x2)/2
            center_y = (y1 + y2)/2
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            if(center_x>330):
                cv2.putText(img, "Kanan", [50,50], font, fontScale, color, thickness)
            elif(center_x<310):
                cv2.putText(img, "Kiri", [50,50], font, fontScale, color, thickness)
            else:
                cv2.putText(img, "center_x", [50,50], font, fontScale, color, thickness)
            
            if(center_y>250):
                cv2.putText(img, "atas", [50,70], font, fontScale, color, thickness)
            elif(center_y<230):
                cv2.putText(img, "bawah", [50,70], font, fontScale, color, thickness)
            else:
                cv2.putText(img, "center_y", [50,70], font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
