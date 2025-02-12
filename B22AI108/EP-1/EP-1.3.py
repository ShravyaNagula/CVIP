import cv2
import numpy as np
image_path= "img2.jpg"
image=cv2.imread(image_path)
rows, cols, _=image.shape
print(image.shape)
translation_matrix=np.float32([[1, 0, 50], [0, 1, 100]])
translated_image=cv2.warpAffine(image, translation_matrix, (cols, rows))
cv2.imshow("Translated image", translated_image)
center=(cols//2, rows//2)
angle=45
scale=1.0
rotation_matrix=cv2.getRotationMatrix2D(center, angle, scale)
rotated_image=cv2.warpAffine(image, rotation_matrix, (cols, rows))
cv2.imshow("Rotated image", rotated_image)
scale_factor = 1.5
scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
cv2.imshow("Scaled Image", scaled_image)
shear_factor_x = 0.5
shear_factor_y = 0.5
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])
sheared_image = cv2.warpAffine(image, shear_matrix, (cols, rows))
cv2.imshow("Sheared Image", sheared_image)
cv2.waitKey(0)
cv2.destroyAllWindows()