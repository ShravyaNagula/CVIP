import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path = 'img_1.png'
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Unable to load the image")
else:
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_image = cv2.filter2D(image, -1, laplacian_kernel)
    threshold = 30
    isolated_points = np.abs(laplacian_image) > threshold
    output_image = image.copy()
    output_image[isolated_points] = 255
    titles = ["Original Image", "Laplacian Convolution", "Isolated Points"]
    images = [image, laplacian_image, output_image]
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')
        plt.axis("off")
    plt.tight_layout()
    plt.show()

