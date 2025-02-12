import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path="img.jpg"
image=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error")
else:
    median=cv2.medianBlur(image, 5)
    kernel=np.ones((3,3),np.uint8)
    minimum=cv2.erode(image, kernel)
    maximum=cv2.dilate(image, kernel)
    titles=["Original image", "Median image", "Minimum image", "maximum image"]
    images=[image, median, minimum, maximum]
    plt.figure(figsize=(12,6))
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title(titles[i])
        plt.imshow(images[i],cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()