import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path="image.jpg     "
image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
height, width=image.shape
sliced_image=np.zeros((height, width), dtype=np.uint8)
min_range=100
max_range=200
for i in range(height):
    for j in range(width):
        if min_range<=image[i,j]<=max_range:
            sliced_image[i,j]=255
        else:
            sliced_image[i,j]=0
titles=["Original image", "Intensity level sliced image"]
images=[image, sliced_image]
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i])
    plt.axis("off")
plt.tight_layout()
plt.show()