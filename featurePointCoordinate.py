from nis import match
import cv2
import numpy as np

# load_image
load_image1 = cv2.imread("../sozai/1.jpg")
load_image2 = cv2.imread("../sozai/2.jpg")

#resize_image
resize_image1 = cv2.resize(load_image1,(800,600))
resize_image2 = cv2.resize(load_image2,(800,600))

#exchange gray_scale -> to enhace processing speed.
gray_image1 = cv2.cvtColor(resize_image1,cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(resize_image2,cv2.COLOR_BGR2GRAY)

#create AKAZA detection 
detector = cv2.AKAZE_create()
kp1,des1 = detector.detectAndCompute(gray_image1,None)
kp2,des2 = detector.detectAndCompute(gray_image2,None)

bf = cv2.BFMatcher()
matches = bf.match(des1,des2)
matches = sorted(matches,key = lambda x:x.distance)
for x in range(len(matches[:20])): 
    print(type(kp1[matches[x].queryIdx].pt))
    print(kp2[matches[x].trainIdx].pt)
    print("---------------------------------------------")


# [:<this>]this argument is feature point figure
img3 = cv2.drawMatches(gray_image1,kp1,gray_image2,kp2,matches[:20],None,flags=2)

cv2.imshow("image2",gray_image2)
cv2.imshow("show",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

