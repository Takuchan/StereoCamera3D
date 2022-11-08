import numpy as np
import cv2
import glob


def calibration(img_dir, board_x, board_y, square_size=1.0):
    objpoint = []
    for x in range(board_x):
        for y in range(board_y):
            objpoint.append([x * square_size, y * square_size, 0])
    objpoint = np.array(objpoint, dtype='float32')

    objpoints = []
    imgpoints = []
    imgs = glob.glob(img_dir + '/*.jpg')

    for img_name in imgs:
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (board_y, board_x), None)
        if ret:
            objpoints.append(objpoint)
            imgpoints.append(corners)
            img = cv2.drawChessboardCorners(img, (board_x, board_y), corners, ret)
            cv2.imwrite('draw.jpg', img)

    ret, k, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('RMS:', ret)
    np.savez('parameters.npz', k=k, dist=dist)
