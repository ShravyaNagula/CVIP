import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path="image.jpg     "
image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Unable to load the image.")
else:
    def histogram_stretching(image):
        min_val=np.min(image)
        max_val=np.max(image)
        stretched=255*(image-min_val)/(max_val-min_val)
        return np.uint8(stretched)
    def histogram_sliding(image,slide_val):
        return np.clip(image,slide_val,0,255).astype(np.uint8)
    def histogram_shrinking(image,shrink_factor):
        center=128
        return np.clip(center+(image-center)/shrink_factor,0,255).astype(np.uint8)
    slid_image=histogram_sliding(image, slide_val=50)
    shrink_image=histogram_shrinking(image, shrink_factor=2)
    stretched_image=histogram_stretching(image)
    def plot_histogram(image,title):
        histogram=cv2.calcHist([image],[0],None,[255],[0,256])
        plt.plot(histogram,label=title)
    plt.figure(figsize=(10,6))
    plt.title("Histogram")
    plt.histogram(image,"Original image")
    plt.histogram(stretched_image,"stretched image")
    plt.histogram(slid_image,"slided image")
    plt.histogram(shrink_image,"shrinked image")