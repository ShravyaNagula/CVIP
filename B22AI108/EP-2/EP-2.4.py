import cv2
import matplotlib.pyplot as plt
image_path="image.jpg"
image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Unable to load the image")
else:
    bit_planes=[]
    for i in range(8):
        bit_plane=(image>>1)&1
        bit_planes.append(bit_plane*255)
    plt.figure(figsize=(12,6))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.title(f"Bit plane {i}")
    plt.imshow(bit_planes[i],cmap="gray")
    plt.axis("off")
cv2.imshow("Image",image)
plt.tight_layout()
plt.show()