import cv2
img_path="img2.jpg"
img=cv2.imread(img_path)
resized_img = cv2.resize(img, (50, 50))
cv2.imshow("img1.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
