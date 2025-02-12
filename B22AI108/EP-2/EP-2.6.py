import cv2
import numpy as np

def histogram_equalization(img):
    # Calculate the histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # Calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalize to [0, 255]

    # Use the CDF to map the pixel values
    img_equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized).reshape(img.shape)

    return img_equalized.astype(np.uint8)

# Load the grayscale image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
img_eq = histogram_equalization(img)

# Display the images
cv2.imshow('Original Image', img)
cv2.imshow('Equalized Image', img_eq)

cv2.waitKey(0)
cv2.destroyAllWindows()
