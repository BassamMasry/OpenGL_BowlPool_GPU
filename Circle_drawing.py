import numpy as np
import cv2

width = 100
height = 100
img = np.zeros(width * height, dtype = np.uint8)
r = 10
numCircles = 3

maxrad = min(width, height) // 2

for i in range (numCircles):
    t = i / (numCircles - 1)
    r = maxrad * (1-t) + 1 / numCircles * t
    print (r)
    color = r * 2
    for x in range (maxrad, width):
        xcord = x - maxrad
        for y in range (maxrad, height):
            ycord = y - maxrad
            if (xcord * xcord + ycord * ycord <= r * r):
                img[y * width + x] = color
                img[y * width + (maxrad - xcord)] = color
                img[(maxrad - ycord) * width + x] = color
                img[(maxrad - ycord) * width + (maxrad - xcord)] = color

IMG = img.reshape(width, height)

cv2.namedWindow("Boba")
cv2.imshow("Boba",IMG)
cv2.waitKey(0)
cv2.destroyAllWindows()
