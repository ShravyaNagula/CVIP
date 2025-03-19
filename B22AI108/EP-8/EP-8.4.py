import cv2
import numpy as np
img_path= "../EP-12.3.25/img_1.png"
image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: unable to load the image")
else:
    _, binary=cv2.threshold(image, 127,255,cv2.THRESH_BINARY)
    kernel=np.ones((3,3),np.uint8)
    thickened_image=cv2.dilate(binary,kernel,iterations=1)
    cv2.imwrite("Thickened_output.png",thickened_image)
    cv2.imshow("Original image",image)
    cv2.imshow("Thickened image",thickened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()