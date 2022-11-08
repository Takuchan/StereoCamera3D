import cv2
import numpy as np


def featurePoints(img_name1, img_name2, width, height, n=10000):
# initialize variable
    image1list = []
    image2list = []

# load_image
    image1 = cv2.imread(img_name1)
    image2 = cv2.imread(img_name2)

# resize_image
    resize_image1 = cv2.resize(image1, (width, height))
    resize_image2 = cv2.resize(image2, (width, height))

# exchange gray_scale -> to enhace processing speed.
    gray_image1 = cv2.cvtColor(resize_image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(resize_image2, cv2.COLOR_BGR2GRAY)

# create AKAZA detection 
    detector = cv2.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(gray_image1, None)
    kp2, des2 = detector.detectAndCompute(gray_image2, None)

    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    for x in range(len(matches[:n])): 
        print(kp1[matches[x].queryIdx].pt)
        print(kp2[matches[x].trainIdx].pt)
        image1list.append(kp1[matches[x].queryIdx].pt)
        image2list.append(kp2[matches[x].trainIdx].pt)
        print("---------------------------------------------")

# [:<this>]this argument is feature point figure
    img3 = cv2.drawMatches(gray_image1, kp1, gray_image2, kp2, matches[:1000], None, flags=2)
    beforeCameraResult = np.array(image1list, dtype=float)
    afterCameraResult = np.array(image2list, dtype=float)
    print("moto-gazo", beforeCameraResult)
    print("ato-gazo", afterCameraResult)
    cv2.imwrite("show.jpg", img3)
    np.savez('featurePoints.npz', x1=beforeCameraResult, x2=afterCameraResult)
