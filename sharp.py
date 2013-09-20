import cv2
import numpy as np

def sharpen(img):
    tmp = cv2.GaussianBlur(img,(5,5),5)
    tmp = cv2.addWeighted(img,1.5,tmp,-0.5,0)
    return tmp

def hist(img):
    h = np.zeros((300,256,3))

    bins = np.arange(256).reshape(256,1)
    color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([img],[ch],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.column_stack((bins,hist))
        cv2.polylines(h,[pts],False,col)
    h=np.flipud(h)
    return h

img = cv2.imread("lena.jpg")
#h1 = hist(img)
cv2.imshow("Original",img)
#cv2.imshow("hist1",h1)

sharped = sharpen(img)
h2 = hist(sharped)
cv2.imshow("Sharped",sharped)
cv2.imshow("hist2",h2)

cv2.waitKey()
cv2.destroyAllWindows()
