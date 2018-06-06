import cv2
import matplotlib.pyplot as plt

image = cv2.imread('center.jpg')
image_filp = cv2.flip(image, 1)
image_crop = image[75:135, 0:320]
cv2.imwrite('center_crop.jpg', image_crop)
plt.imshow(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
plt.show()
