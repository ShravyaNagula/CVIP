import cv2
import numpy as np
image_path = 'img_1.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
seed = (100, 100)
threshold = 20
segmented = np.zeros_like(img)
seed_intensity = img[seed[1], seed[0]]
to_check = [seed]
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
while to_check:
    x, y = to_check.pop()
    if segmented[y, x] == 1:
        continue
    segmented[y, x] = 1
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0]:
            if abs(int(img[ny, nx]) - int(seed_intensity)) <= threshold and segmented[ny, nx] == 0:
                to_check.append((nx, ny))
cv2.imshow('Segmented Image', segmented * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
