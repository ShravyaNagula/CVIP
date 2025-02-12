import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path = "image.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
def negative_transformation(image):
    return 255 - image
def log_transformation(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * np.log(1 + image.astype(np.float32))
    return np.uint8(log_image)
def gamma_transformation(image, gamma):
    return np.array(255 * (image / 255) ** gamma, dtype=np.uint8)
negative_image = negative_transformation(image)
log_image = log_transformation(image)
gamma_image = gamma_transformation(image, gamma=2.0)
titles = ["Original image", "Negative image", "Log image", "Gamma image"]
images = [image, negative_image, log_image, gamma_image]
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.show()

