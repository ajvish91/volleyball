# ##################################################################

# # Your code goes here

# ##################################################################

# import relevant packages
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt

# capture video
cap = cv2.VideoCapture('media/beachVolleyball7.mov')
# get video information
width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CV_CAP_PROP_FPS)
frameCount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
print width, height, fps, frameCount

# extract frames from the video
_, img = cap.read()
avgImg = np.float32(img)
print img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eql_gray = cv2.equalizeHist(gray)
corner_image = cv2.cornerHarris(eql_gray, 2, 3, 0.05)
print corner_image
img[corner_image > 0.03 * corner_image.max()] = [0, 0, 255]
cv2.imwrite("corners.jpg", img)

# go through each frame
for fr in range(1, frameCount):
    alpha = 1.0 / (fr + 1)
    _, img = cap.read()
    cv2.accumulateWeighted(img, avgImg, alpha)
    normImg = cv2.convertScaleAbs(avgImg)
    cv2.imshow('img', img)
    cv2.imshow('normImg', normImg)
    cv2.imwrite('normImg.jpg', normImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
