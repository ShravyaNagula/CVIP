import cv2
import numpy as np
image=cv2.imread("../EP-10/img.png")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,threshold1=50,threshold2=150)
lines=cv2.HoughLinesP(edges,rho=1,theta=np.pi/180,threshold=100,minLineLength=50,maxLineGap=10)
if lines is not None:
    for line in lines:
        x1,y1,x2,y2=line[0]
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imwrite("Detected_Lines.png",image)
cv2.imshow("Line Detection",image)