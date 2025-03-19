import cv2
import numpy as np
img_path = "../EP-12.3.25/img_1.png"
image = cv2.imread(img_path)
if image is None:
    print("Error: Could not read the image.")
    exit()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hull_list = [cv2.convexHull(cnt) for cnt in contours]
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.drawContours(image, hull_list, -1, (0, 0, 255), 2)
cv2.imshow("Convex Hulls and Contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

