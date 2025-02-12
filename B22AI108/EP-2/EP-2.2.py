import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path="image.jpg"
image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Unable to load the image. please check the file!")
else:
    def contrast_stretch(image):
        min_val=np.min(image)
        max_val=np.max(image)
        stretched_image=255*(image-min_val)/(max_val-min_val)
        return np.uint8(stretched_image)
stretched_image=contrast_stretch(image)
titles=["Original image", "Stretched image"]
images=[image, stretched_image]
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(titles[i])
    plt.imshow(images[i])
    plt.axis("off")
plt.tight_layout()
plt.show()