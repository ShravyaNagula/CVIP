import cv2
img_path="image.jfif"
image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Could not load image.")
else:
    mean_filtered = cv2.blur(image, (5, 5))
    gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow('Original Image', image)
    cv2.imshow('Mean Filtered Image', mean_filtered)
    cv2.imshow('Gaussian Filtered Image', gaussian_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()