import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

def find_nRound(row):
    dst = 512.0/row
    n = 0
    cont = 2/3.0
    val = cont
    while val>dst:
        n += 1
        val *= cont
    if n == 0:
        return 0
    else:
        return n+1

def shapen_n(img):
    
    row = img.shape[0]
    n = find_nRound(row)
     
    n = n*2+3
    sharped = img.copy()
    print "n=%d"%n
    for i in range(n,3,-2):
        tmp = cv2.GaussianBlur(sharped,(i,i),5)
        sharped = cv2.addWeighted(sharped,1.5,tmp,-0.5,0)
        sharped = cv2.resize(sharped, (0,0), fx=2/3.0, fy=2/3.0) 
    
    tmp = cv2.GaussianBlur(sharped,(3,3),5)
    sharped = cv2.addWeighted(sharped,1.5,tmp,-0.5,0)
    return sharped


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

def spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
   
    return magnitude_spectrum

filename = "lena.jpg"
if len(sys.argv) >= 2:
    filename = sys.argv[1]

img = cv2.imread(filename)
row , col , chan = img.shape

sharped = sharpen(img)

sharped_n = shapen_n(img)
if row > 512:
    while row > 512 :
        sharped = cv2.resize(sharped, (0,0), fx=2/3.0, fy=2/3.0) 
        img = cv2.resize(img, (0,0), fx=2/3.0, fy=2/3.0) 
        sharped = sharpen(sharped)
        row , col , chan = sharped.shape
       
cv2.imshow("Original",img)
h1 = hist(img)
cv2.imshow("hist1",h1)

cv2.imshow("Sharped",sharped)
h2 = hist(sharped)
cv2.imshow("hist2",h2)

cv2.imshow("Sharped_n",sharped_n)
h3 = hist(sharped_n)
cv2.imshow("hist3",h3)

#display spectrum
m3 = spectrum(sharped_n)
m1 = spectrum(img)
m2 = spectrum(sharped)

plt.subplot(131),plt.imshow(m1, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(m2, cmap = 'gray')
plt.title('Shapen'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(m3, cmap = 'gray')
plt.title('Shapen_n'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.imwrite("__"+filename,sharped)
cv2.imwrite("_n_"+filename,sharped_n)
cv2.waitKey()
cv2.destroyAllWindows()
