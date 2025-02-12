import cv2
img_path = "img2.jpg"
img_colour = cv2.imread(img_path)
gray_img = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)
_, img_bin = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Original Image", img_colour)
img_colour.size
cv2.imshow("Gray Image", gray_img)
cv2.imshow("Binary Image", img_bin)
cv2.waitKey(0)
cv2.destroyAllWindows()