import cv2
import numpy as np
img_path= "../EP-12.3.25/img_1.png"
image=cv2.imread(img_path)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
sift=cv2.SIFT_create()
keypoints=sift.detect(gray,None)
output_image=cv2.drawKeypoints(image,keypoints,None,color=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Original image",image)
cv2.imshow("SIFT keypoints",output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()