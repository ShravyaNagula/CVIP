import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
image_path="image.jfif"
image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error")
else:
    image=cv2.normalize(image,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    mean=0
    stddev=25
    gaussian_noise=np.random.normal(mean, stddev,image.shape).astype(np.uint8)
    noisy_image=cv2.add(image, gaussian_noise)
    noisy_image = cv2.normalize(noisy_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    restored_image=wiener(noisy_image,(5,5))
    restored_image=np.clip(restored_image,0,255).astype(np.uint8)
    titles=["Original image","Noisy image","Restored image"]
    images=[image,noisy_image,restored_image]
    plt.figure(figsize=(12,8))
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.title(titles[i])
        plt.imshow(images[i],cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()