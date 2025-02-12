import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path="images.jfif"
image=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error")
else:
    kernel=np.array([[0, -1, 0],
                     [-1, 4, -1],
                     [0, -1, 0]])
    sharpened=cv2.filter2D(image, -1, kernel)
    titles=["Original image", "Sharpened image"]
    images=[image, sharpened]
    plt.figure(figsize=(10, 5))
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()