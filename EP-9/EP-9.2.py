import cv2
img_path= "../EP-12.3.25/img_1.png"
image=cv2.imread(img_path)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray, threshold1=100, threshold2=200)
cv2.imshow("Original image",image)
cv2.imshow("Point detection using Canny algorithm",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()