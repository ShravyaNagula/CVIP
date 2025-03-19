import cv2
import numpy as np
cap=cv2.videoCapture(0)
if not cap.Opened():
    print("Error")
else:
    while True:
        ret, frame=cap.read()
        if not ret:
            print("Error")
            break
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lower_bound=np.array([30,50,50])
        upper_bound=np.array([90,255,255])
        mask=cv2.inRange(hsv, lower_bound, upper_bound)
        result=cv2.bitwise_and(frame,frame,mask=mask)
        kernel=np.ones((5,5),np.uint8)
        opening=cv2.morphologyEX(mask,cv2.MORPH_OPEN,kernel)
        closing=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        cv2.imshow("original Frame",frame)
        cv2.imshow("Mask",mask)
        cv2.imshow("Opening Operation",opening)
        cv2.imshow("Closing Operation",closing)
cv2.waitKey(0)
cv2.destroyAllWindows()