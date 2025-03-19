import cv2
import numpy as np
img_path= "../EP-10/img.png"
image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Unable tho load the image.")
else:
    kernel=np.ones((5,5),np.uint8)
    erode_image=cv2.erode(image,kernel,iterations=1)
    boundary_image=image-erode_image
    cv2.imshow("Original image",image)
    cv2.imshow("Boundary image",boundary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()