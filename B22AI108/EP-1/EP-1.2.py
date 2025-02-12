import cv2
img_path= "img2.jpg"
img_colour=cv2.imread(img_path)
gray_img=cv2.cvtColor(img_colour,cv2.COLOR_BGR2GRAY)
cv2.imshow("Original Image",img_colour)
cv2.imshow("Gray Image",gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()