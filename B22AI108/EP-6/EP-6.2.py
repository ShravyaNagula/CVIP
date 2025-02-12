import cv2
import numpy as np
img_path="image.jfif"
image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error")
else:
    mean=0
    std_dev=25
    gaussian_noise=np.random.normal(mean,std_dev,image.shape).astype(np.uint8)
    noisy_img=cv2.add(image,gaussian_noise)
    noisy_img=cv2.normalize(noisy_img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("Original image",image)
    cv2.imshow("Noisy image",noisy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()