import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path="image.jfif"
image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Could not load image.")
else:
    def add_gaussian_noise(image, mean=0, std=25):
        noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image
    def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
        noisy_image = np.copy(image)
        salt_mask = np.random.uniform(0, 1, image.shape) < salt_prob
        noisy_image[salt_mask] = 255
        pepper_mask = np.random.uniform(0, 1, image.shape) < pepper_prob
        noisy_image[pepper_mask] = 0
        return noisy_image
    gaussian_noisy_image = add_gaussian_noise(image, mean=0, std=30)
    salt_pepper_noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(gaussian_noisy_image, cmap='gray')
    plt.title('Image with Gaussian Noise')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(salt_pepper_noisy_image, cmap='gray')
    plt.title('Image with Salt-and-Pepper Noise')
    plt.axis('off')
plt.tight_layout()
plt.show()