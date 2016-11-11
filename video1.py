import cv2
import cv2.cv as cv
import numpy as np
import time

cap = cv2.VideoCapture('media/beachVolleyball1.mov')
FRAME_COUNT = int(cap.get(7))
print FRAME_COUNT
FRAME_INTERVAL = 1
#------------ manually chosen features
FRAME_START = 0
FRAME_END = 740
FULL_COURT_WIDTH = 632
FULL_COURT_HEIGHT = 300
CORNER_1 = [199, 62]
CORNER_2 = [48, 88]
CORNER_3 = [441, 137]
CORNER_4 = [203, 287]
p0 = np.array([[120.0,180.0],[345.0,202.0],[295.0,81.0],[250.0,220.0],[45.0,145.0]], np.float32)
p_dst = np.array([[120.0,180.0],[345.0,202.0],[295.0,81.0],[250.0,220.0],[45.0,145.0]], np.float32)
#------------ manually chosen features

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))
# Create some random colors for lk track
color = np.random.randint(0,255,(100,3))


# First frame
ret, old_frame = cap.read()
rows,cols,channels = old_frame.shape
#print 'rows', rows
#print 'cols', cols
h, status = cv2.findHomography(p0, p_dst)
old_frame = old_frame[0:rows, 0:cols-5]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
im_out = cv2.warpPerspective(old_frame, h, (FULL_COURT_WIDTH, FULL_COURT_HEIGHT))
cv2.imshow('frame',im_out)

mask_draw = np.zeros_like(old_frame)
arr = []
arr.append(im_out.copy())
fr = FRAME_START
avgImg = np.float32(im_out)
fr_counter = 1

#Background generator
while(fr < FRAME_END):
    ret,frame = cap.read()
    rows,cols,channels = frame.shape
    frame = frame[0:rows, 0:cols-5]
    if (fr_counter%FRAME_INTERVAL == 0):

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        h, status = cv2.findHomography(p1, p_dst)
        im_out_new = cv2.warpPerspective(frame, h, (FULL_COURT_WIDTH,FULL_COURT_HEIGHT))

        rows,cols,channels = im_out_new.shape
        roi = im_out[0:rows, 0:cols]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(im_out_new,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(im_out_new,im_out_new,mask = mask)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg,img2_fg)
        im_out[0:rows, 0:cols] = dst
        im_gray = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
        arr.append(im_out.copy())
        cv2.imshow('frame',im_out)

        # getting the background
        alpha = 1/float(fr+1)
        cv2.accumulateWeighted(im_out,avgImg,alpha)
        im_out = cv2.convertScaleAbs(avgImg)
        normImg = im_out.copy()

        cv2.imshow('background',normImg)

        # draw the tracks
        for i,(new,old) in enumerate(zip(p1,p0)):
            a,b = new.ravel()
            c,d = old.ravel()
            cv2.line(mask_draw, (a,b),(c,d), color[i].tolist(), 2)
            cv2.circle(frame,(a, b),3,color[i].tolist(),-1)
        img = cv2.add(frame,mask_draw)

        cv2.imshow('tracking',img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        old_frame = frame.copy()
        p0 = p1.copy()
        cv2.waitKey(1)
    fr_counter += 1
    fr = fr + 1

cv2.imwrite('back1.jpg', normImg)
# Background Substraction

bg_gray = cv2.cvtColor(normImg, cv2.COLOR_BGR2GRAY)
p_bg = np.array([[140.0,210.0],[90.0,316.0],[743.0,187.0],[800.0,282.0]], np.float32)

court_top = cv2.imread('court.png',1)
p_top = np.array([[40.0,115.0],[40.0,350.0],[515.0,115.0],[515.0,350.0]], np.float32)
h_top, status = cv2.findHomography(p_bg, p_top)
print "homography top", h_top

bg_gray = cv2.GaussianBlur(bg_gray, (21, 21), 0)
for fra_ori in arr:
    fra = cv2.cvtColor(fra_ori, cv2.COLOR_BGR2GRAY)
    fra = cv2.GaussianBlur(fra, (21, 21), 0)
    frame_delta = cv2.subtract(bg_gray, fra)
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=6)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    court_top_copy = court_top.copy()
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 5 and cv2.contourArea(c) > 10:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(fra_ori, (x, y), (x + w, y + h), (0, 255, 0), 2)

        xi = (x+x+h)/2
        yi = y+h

        h0 = h_top[0,0]
        h1 = h_top[0,1]
        h2 = h_top[0,2]
        h3 = h_top[1,0]
        h4 = h_top[1,1]
        h5 = h_top[1,2]
        h6 = h_top[2,0]
        h7 = h_top[2,1]
        h8 = h_top[2,2]

        tx = (h0*xi + h1*yi + h2)
        ty = (h3*xi + h4*yi + h5)
        tz = (h6*xi + h7*yi + h8)

        px = int(tx/tz)
        py = int(ty/tz)
        #Print the player on top down view
        cv2.circle(court_top_copy,(px, py),3,(0,0,255),-1)

    #cv2.imshow('frame_delta',frame_delta)
    #cv2.imshow('frame_thresh',thresh)
    cv2.imshow('frame_detect',fra_ori)
    cv2.imshow('court_top', court_top_copy)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
