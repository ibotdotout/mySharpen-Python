import cv2
import numpy as np

img = cv2.imread("lena.jpg")
cv2.imshow("Original",img)
cv2.waitKey()
cv2.destroyAllWindows()
