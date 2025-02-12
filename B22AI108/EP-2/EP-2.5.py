import cv2
import matplotlib.pyplot as plt
image_path="image.jpg"
image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Unable to load the image")
else:
    histogram=cv2.calcHist([image],[0],None,[255],[0,256])
    plt.figure(figsize=(8,6))
    plt.title("GrayScale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(histogram, color='black')
    plt.xlim([0,256])
    plt.tight_layout()
    plt.show()