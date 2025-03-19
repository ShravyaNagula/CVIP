import cv2
import numpy as np
img_path= "../EP-12.3.25/img_1.png"
image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
_, binary=cv2.threshold(image,127,255,cv2.THRESH_BINARY)
kernel=np.ones((3,3),np.uint8)
skeleton=np.zeros(binary.shape,dtype=np.uint8)
while True:
    eroded=cv2.erode(binary,kernel)
    temp=cv2.dilate(eroded,kernel)
    skeleton_part=cv2.subtract(binary,temp)
    skeleton=cv2.bitwise_or(skeleton,skeleton_part)
    binary=eroded.copy()
    if cv2.countNonZero(binary)==0:
        cv2.imwrite("Skeleton.png",skeleton)
        cv2.imshow("Original image",image)
        cv2.imshow("Skeleton image",skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()
