import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path = "../EP-4/images.jfif"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error")
else:
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
    rows, cols = image.shape
    crows, ccols = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    radius = 50
    cv2.circle(mask, (ccols, crows), radius, 1, thickness=-1)
    low_pass = dft_shift * mask
    low_pass_ift = np.fft.ifftshift(low_pass)
    low_pass_img = np.abs(np.fft.ifft2(low_pass_ift))
    high_pass = dft_shift * (1 - mask)
    high_pass_ift = np.fft.ifftshift(high_pass)
    high_pass_img = np.abs(np.fft.ifft2(high_pass_ift))
    titles = ["Original image", "Magnitude spectrum image", "Low pass filtered image", "High pass filtered image"]
    images = [image, magnitude_spectrum, low_pass_img, high_pass_img]
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')
        plt.axis("off")
    plt.tight_layout()
    plt.show()
