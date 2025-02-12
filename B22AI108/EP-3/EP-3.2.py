import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path="img.jpg"
image=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Unable to load the image")
else:
    kernel=np.array([1, -1])
    horizontal_convolution=cv2.filter2D(image, -1, kernel.reshape(1, -1))
    vertical_convolution=cv2.filter2D(image, -1, kernel.reshape(-1,1))
    titles=["Original Image", "Horizontal Convolution", "Vertical Convolution"]
    images=[image, horizontal_convolution, vertical_convolution]
    plt.figure(figsize=(12,6))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(titles[i])
        plt.imshow(images[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()