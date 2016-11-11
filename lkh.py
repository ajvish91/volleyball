import cv2
import cv2.cv as cv
import numpy as np
import time

cap = cv2.VideoCapture('beachVolleyball3.mov')

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
rows,cols,channels = old_frame.shape
old_frame = old_frame[0:rows, 0:cols-5]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create a blank image for panoramic (height, width)
blank_image = np.zeros((400,1200,3), np.uint8)

CAP_PROP_FRAME_COUNT = 7
frame_count = int(cap.get(CAP_PROP_FRAME_COUNT))
print frame_count
# Brute force the court points (width, height) for video 3
#p0 = np.array([[160.0,195.0], [160.0,205.0], [160.0,225.0], [160.0,235.0]], np.float32)

# Brute force the court points (width, height) for video 3
p0 = np.array([[160.0,200.0],[235.0,272.0],[310.0,260.0],[250.0,245.0],[300.0,165.0]], np.float32)
p_dst = np.array([[460.0,200.0],[535.0,272.0],[610.0,260.0],[550.0,245.0],[600.0,165.0]], np.float32)

#p0 = np.array([[160.0,200.0],[235.0,272.0],[300.0,260.0],[238.0,145.0],[300.0,165.0]], np.float32)
#p_dst = np.array([[360.0,200.0],[435.0,272.0],[500.0,260.0],[438.0,145.0],[500.0,165.0]], np.float32)

#p_dst = np.array([[212.0,152.0],[148.0,272.0],[520.0,162.0],[512.0,292.0]], np.float32)

#p0 = np.array([[248.0,272.0], [430.0,282.0],[600.0,162.0],[612.0,292.0]], np.float32)
#p_dst = np.array([[248.0,272.0], [430.0,282.0],[600.0,162.0],[612.0,292.0]], np.float32)

counter = 1

h, status = cv2.findHomography(p0, p_dst)
im_out = cv2.warpPerspective(old_frame, h, (1000,350))
cv2.imshow('frame',im_out)

print "points", p0, p0.shape

# Create a mask image for drawing purposes
mask_draw = np.zeros_like(old_frame)

arr = []
old_abu = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
arr.append(old_abu)
fr = 0

#_,img = cap.read()
avgImg = np.float32(im_out)

while(fr < 330):
    ret,frame = cap.read()
    rows,cols,channels = frame.shape
    frame = frame[0:rows, 0:cols-5]
    if (counter%2==0):

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        h, status = cv2.findHomography(p1, p_dst)
        im_out_new = cv2.warpPerspective(frame, h, (1000,350))

        rows,cols,channels = im_out_new.shape
        roi = im_out[0:rows, 0:cols ]

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

        frame_abu = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
        arr.append(frame_abu)
        cv2.imshow('frame',im_out)

        #Background substraction
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

        #cv2.imshow('other',img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        old_frame = frame.copy()
        p0 = p1.copy()
        cv2.waitKey(1)
    counter= counter + 1
    fr = fr + 1

bg_gray = cv2.cvtColor(normImg, cv2.COLOR_BGR2GRAY)
print "bg_gray", bg_gray.shape
print arr[0].shape
for fra in arr:
    #frameDelta = cv2.absdiff(bg_gray, fra)
    cv2.imshow('frame_delta',fra)
    cv2.imshow('frame_bg_gray',bg_gray)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
