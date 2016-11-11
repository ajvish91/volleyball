import cv2
import cv2.cv as cv
import numpy as np
import time

cap = cv2.VideoCapture('media/beachVolleyball2.mov')

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
#blank_image = np.zeros((400,1200,3), np.uint8)

CAP_PROP_FRAME_COUNT = 7
frame_count = int(cap.get(CAP_PROP_FRAME_COUNT))
print frame_count
# Brute force the court points (width, height) for video 3
#p0 = np.array([[160.0,195.0], [160.0,205.0], [160.0,225.0], [160.0,235.0]], np.float32)

# Brute force the court points (width, height) for video 3
#p0 = np.array([[160.0,200.0],[235.0,272.0],[310.0,260.0],[250.0,245.0],[300.0,165.0]], np.float32)
#p_dst = np.array([[460.0,200.0],[535.0,272.0],[610.0,260.0],[550.0,245.0],[600.0,165.0]], np.float32)

#points for video 1
#p0 = np.array([[120.0,180.0],[345.0,202.0],[295.0,81.0],[250.0,220.0],[45.0,145.0]], np.float32)
#p_dst = np.array([[120.0,180.0],[345.0,202.0],[295.0,81.0],[250.0,220.0],[45.0,145.0]], np.float32)

#points for video 2
p0 = np.array([[139.0,91.0],[316.0,70.0],[375.0,199.0],[555.0,150.0],[566.0,110.0]], np.float32)
p_dst = np.array([[139.0,91.0],[316.0,70.0],[375.0,199.0],[555.0,150.0],[566.0,110.0]], np.float32)

#points for video 2 2ndpart
p0n = np.array([[99.0,177.0],[569.0,147.0],[505.0,127.0],[575.0,196.0],[123.0,223.0]], np.float32)
p_dstn = np.array([[210.0,109.0],[576.0,91.0],[526.0,76.0],[580.0,128.0],[240.0,148.0]], np.float32)

#points for video 2 3rdpart
p0n2 = np.array([[216.0,90.0],[210.0,187.0],[599.0,101.0],[603.0,139.0],[498.0,174.0]], np.float32)
p_dstn2 = np.array([[195.0,81.0],[184.0,180.0],[575.0,90.0],[580.0,128.0],[477.0,166.0]], np.float32)

#points for video 4 1st part
#p0 = np.array([[550,110.0],[403.0,205.0],[570.0,280.0],[369.0,248.0],[622.0,199.0]], np.float32)
#p_dst = np.array([[550,110.0],[403.0,205.0],[570.0,280.0],[369.0,248.0],[629.0,199.0]], np.float32)

#points for video 5
#p0 = np.array([[225.0,152.0],[180.0,189.0],[400.0,200.0],[467,190.0],[188.0, 165.0]], np.float32)
#p_dst = np.array([[225.0,202.0],[180.0,239.0],[400.0,250.0],[467,240.0],[188.0, 215.0]], np.float32)

#points for video 6
#p0 = np.array([[402.0,270.0],[600.0,300.0],[370.0,210.0],[467,170.0],[550.0, 250.0]], np.float32)
#p_dst = np.array([[402.0,320.0],[600.0,350.0],[370.0,260.0],[467,220.0],[550.0, 300.0]], np.float32)

#points for video 7
#p0 = np.array([[10.0,185.0],[216.0,330.0],[165.0,336.0],[240,200.0],[65.0, 198.0]], np.float32)
#p_dst = np.array([[310.0,185.0],[516.0,330.0],[465.0,336.0],[540,200.0],[365.0, 198.0]], np.float32)

#p0 = np.array([[160.0,200.0],[235.0,272.0],[300.0,260.0],[238.0,145.0],[300.0,165.0]], np.float32)
#p_dst = np.array([[360.0,200.0],[435.0,272.0],[500.0,260.0],[438.0,145.0],[500.0,165.0]], np.float32)

#p_dst = np.array([[212.0,152.0],[148.0,272.0],[520.0,162.0],[512.0,292.0]], np.float32)

#p0 = np.array([[248.0,272.0], [430.0,282.0],[600.0,162.0],[612.0,292.0]], np.float32)
#p_dst = np.array([[248.0,272.0], [430.0,282.0],[600.0,162.0],[612.0,292.0]], np.float32)

counter = 1

h, status = cv2.findHomography(p0, p_dst)
im_out = cv2.warpPerspective(old_frame, h, (1000,300))
cv2.imshow('frame',im_out)

# Create a mask image for drawing purposes
mask_draw = np.zeros_like(old_frame)

arr = []
old_abu = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
#arr.append(old_abu)
arr.append(im_out.copy())
fr = 0

#_,img = cap.read()
avgImg = np.float32(im_out)

while(fr < 150):
    ret,frame = cap.read()
    rows,cols,channels = frame.shape
    frame = frame[0:rows, 0:cols-5]
    if (counter%1==0):

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        h, status = cv2.findHomography(p1, p_dst)
        im_out_new = cv2.warpPerspective(frame, h, (1000,300))

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
    counter= counter + 1
    fr = fr + 1
#cv2.waitKey(0)
while(fr < 365):
    ret,frame = cap.read()
    rows,cols,channels = frame.shape
    frame = frame[0:rows, 0:cols-5]
    if (counter%1==0):

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0n, None, **lk_params)

        h, status = cv2.findHomography(p1, p_dstn)
        im_out_new = cv2.warpPerspective(frame, h, (632,300))

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
        p0n = p1.copy()
        cv2.waitKey(1)
    counter= counter + 1
    fr = fr + 1
#cv2.waitKey(0)
while(fr < 570):
    ret,frame = cap.read()
    rows,cols,channels = frame.shape
    frame = frame[0:rows, 0:cols-5]
    if (counter%1==0):

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0n2, None, **lk_params)

        h, status = cv2.findHomography(p1, p_dstn2)
        im_out_new = cv2.warpPerspective(frame, h, (632,300))

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
        p0n2 = p1.copy()
        cv2.waitKey(1)
    counter= counter + 1
    fr = fr + 1

# Background Substraction
#cv2.imwrite('bg_3.png',normImg)
bg_gray = cv2.cvtColor(normImg, cv2.COLOR_BGR2GRAY)
p_bg = np.array([[0, 191],[595, 264],[347, 82],[575, 89]], np.float32)

court_top = cv2.imread('court.png',1)
p_top = np.array([[40.0,115.0],[40.0,350.0],[515.0,115.0],[515.0,350.0]], np.float32)
h_top, status = cv2.findHomography(p_bg, p_top)
print "homography top", h_top

old_panorama = arr[0]
old_panorama_gray = cv2.cvtColor(old_panorama, cv2.COLOR_BGR2GRAY)
p_player_0 = np.array([[140.0,268.0],[422.0,186.0],[505.0,90.0],[382.0,85.0]], np.float32)
mask_draw = np.zeros_like(old_panorama)

#bg_gray = cv2.GaussianBlur(bg_gray, (21, 21), 0)
for fra_ori in arr:
    fra_gray = cv2.cvtColor(fra_ori, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p_player_1, st, err = cv2.calcOpticalFlowPyrLK(old_panorama_gray, fra_gray, p_player_0, None, **lk_params)

    court_top_copy = court_top.copy()
    for i,(new,old) in enumerate(zip(p_player_1,p_player_0)):
        xi,yi = new.ravel()
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask_draw, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(fra_ori,(a, b),3,color[i].tolist(),-1)

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
    img = cv2.add(fra_ori,mask_draw)
    #fra = cv2.GaussianBlur(fra, (21, 21), 0)

    #cv2.imshow('frame_delta',frame_delta)
    #cv2.imshow('frame_thresh',thresh)
    cv2.imshow('frame_detect',fra_ori)
    cv2.imshow('court_top', court_top_copy)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    old_panorama_gray = fra_gray.copy()
    old_panorama = fra_ori.copy()
    p_player_0 = p_player_1.copy()

cv2.destroyAllWindows()
cap.release()
