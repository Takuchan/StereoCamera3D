import cv2
import numpy as np


# 内部パラメーター: k1, k2
# 特徴点: x1, x2
def calculate3dPoint(k1, k2, x1, x2):
    F = cv2.findFundamentalMat(x1, x2)[0]
    E = k1 @ F @ k2
    R1, R2, t = cv2.decomposeEssentialMat(E)
    _, R, t, _ = cv2.recoverPose(E, x1, x2)
    cameraMat1 = k1 @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    cameraMat2 = k2 @ np.concatenate([R, t], 1)
    p = cv2.triangulatePoints(cameraMat1, cameraMat2, np.transpose(x1), np.transpose(x2))
    return p / p[3]
