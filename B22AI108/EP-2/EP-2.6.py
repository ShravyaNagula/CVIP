import cv2
import numpy as np

def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalize to [0, 255]
    img_equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized).reshape(img.shape)
    return img_equalized.astype(np.uint8)
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
img_eq = histogram_equalization(img)
cv2.imshow('Original Image', img)
cv2.imshow('Equalized Image', img_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()
