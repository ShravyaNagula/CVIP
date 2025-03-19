import cv2
import numpy as np
img_path= "../EP-8/img1.jpg"
image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error")
else:
    kernel=np.array([[0,1,0],
                     [1,-1,1],
                     [0,1,0]],dtype=np.int8)
    hit_or_miss=cv2.morphologyEx(image,cv2.MORPH_HITMISS,kernel)
    cv2.imshow("Original image",image)
    cv2.imshow("Hit-or-Miss image",hit_or_miss)
cv2.waitKey(0)
cv2.destroyAllWindows()