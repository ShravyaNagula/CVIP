import cv2
import numpy as np
img_path= "../EP-12.3.25/img_1.png"
image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Unable tho load the image.")
else:
    kernel=np.ones((5,5),np.uint8)
    dilate_image=cv2.dilate(image,kernel,iterations=1)
    boundary_image=dilate_image-image
    cv2.imshow("Original image",image)
    cv2.imshow("Outer Boundary image",boundary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()