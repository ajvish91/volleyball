import cv2
import cv2.cv as cv
import numpy as np
import time

cap = cv2.VideoCapture('beachVolleyball4.mov')

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create a blank image for panoramic (height, width)
blank_image = np.zeros((400,1200,3), np.uint8)

# Brute force the court points (width, height) for video 3
#p0 = np.array([[160.0,195.0], [160.0,205.0], [160.0,225.0], [160.0,235.0]], np.float32)

# Brute force the court points (width, height) for video 4
p0 = np.array([[312.0,152.0],[248.0,272.0],[620.0,162.0],[612.0,292.0]], np.float32)
p_dst = np.array([[212.0,152.0],[148.0,272.0],[520.0,162.0],[512.0,292.0]], np.float32)

h, status = cv2.findHomography(p0, p_dst)
im_out = cv2.warpPerspective(old_gray, h, (1200,400))
cv2.imshow('frame',im_out)

print "points", p0, p0.shape

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    h, status = cv2.findHomography(p1, p_dst)
    im_out = cv2.warpPerspective(frame_gray, h, (1200,400))
    #cv2.imshow('frame',im_out)

    # draw the tracks
    for i,(new,old) in enumerate(zip(p1,p0)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(frame,(a, b),3,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = p1.copy()
    cv2.waitKey(20)

cv2.destroyAllWindows()
cap.release()
