import cv2


def equalizeBGRImage(img):
    img_hsv = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
    eql_img = cv2.equalizeHist(img_hsv[:, :, 2])
    # print eql_img.shape
    img_hsv[:, :, 2] = eql_img[:, :]
    return cv2.cvtColor(img_hsv, cv2.cv.CV_HSV2BGR)
