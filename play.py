# ##################################################################

# # Your code goes here

# ##################################################################

# import relevant packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

# capture video
cap = cv2.VideoCapture('media/beachVolleyball7.mov')
# get video information
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print width, height, fps, frameCount

# extract frames from the video
_, img = cap.read()
avgImg1 = np.float32(img)
avgImg2 = np.float32(img)
normImg1 = np.empty(img.shape)
normImg2 = np.empty(img.shape)
normImg = np.empty([356, 1000, 3])
print img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eql_gray = cv2.equalizeHist(gray)

# go through each frame
for fr in range(1, frameCount):
    _, img = cap.read()
    if 650 < fr and fr < 700:
        alpha = 1.0 / (fr - 650 + 1)
        cv2.accumulateWeighted(img, avgImg1, alpha)
        normImg1 = cv2.convertScaleAbs(avgImg1)
        # cv2.imshow('norm img2', normImg1)
    if 950 < fr:
        alpha = 1.0 / (fr - 950 + 1)
        cv2.accumulateWeighted(img, avgImg2, alpha)
        normImg2 = cv2.convertScaleAbs(avgImg2)
        # cv2.imshow('norm img1', normImg2)
color = np.array([255, 0, 255])
p1 = np.array([[490.0, 140.0], [480.0, 184.0], [
              380.0, 133.0], [380.0, 183.0]], np.float32)
p2 = np.array([[210.0, 157.0], [200.0, 201.0], [100.0, 150.0], [
              100.0, 200.0]], np.float32)
for i, (new, old) in enumerate(zip(p1, p2)):
    a, b = new.ravel()
    c, d = old.ravel()
    cv2.circle(normImg1, (a, b), 3, color, -1)
    cv2.circle(normImg2, (c, d), 3, color, -1)

cv2.imshow('img 2', normImg2)
cv2.imshow('img 1', normImg1)
h, status = cv2.findHomography(p2, p1)
im_out = cv2.warpPerspective(normImg2, h, (1000, 400))
cv2.imshow('frame', im_out)

# cv2.imwrite('normImpg', normImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
