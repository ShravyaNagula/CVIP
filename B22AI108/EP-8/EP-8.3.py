import cv2
import numpy as np
from skimage.morphology import skeletonize
img_path= "../EP-12.3.25/img_1.png"
image=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: unable to load the image")
else:
    _, binary=cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    binary=binary//255
    skeleton=skeletonize(binary)
    skeleton=(skeleton*255).astype(np.uint8)
    cv2.imshow("Original image",image)
    cv2.imshow("Thinned image",skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()