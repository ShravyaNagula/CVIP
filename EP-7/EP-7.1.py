import cv2
import numpy as np
import matplotlib.pyplot
img_path="img1.jpg, "
image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: image not found")
else:
    kernel=np.ones((5,5),np.uint8)
    enr=cv2.erode(image, kernel, iterations=1)
    dil=cv2.dilate(image, kernel, iterations=1)
    cv2.imshow("Original image",image)
    cv2.imshow("Eroded image",enr)
    cv2.imshow("Dilated image",dil)
cv2.waitKey(0)
cv2.destroyAllWindows()