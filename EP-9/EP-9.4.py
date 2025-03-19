import cv2
image=cv2.imread("../EP-12.3.25/img_1.png")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,threshold1=100,threshold2=200)
cv2.imwrite("Edge detection.png",edges)
cv2.imshow("Original image",image)
cv2.imshow("Edge detection using Canny",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()