import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path="img.jpg"
image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error")
else:
    gaussian_blurred=cv2.GaussianBlur(image,(5,5),0)
    box_filtered=cv2.blur(image,(5,5))
    median_filtered=cv2.medianBlur(image,5)
    titles=["Original image","Gaussian image","Box image","Median image"]
    images=[image, gaussian_blurred, box_filtered, median_filtered]
    plt.figure(figsize=(12,6))
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title(titles[i])
        plt.imshow(images[i],cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()