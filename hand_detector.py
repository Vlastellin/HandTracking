from __future__ import division
import cv2
import numpy as np

protoFile = "model/pose_deploy.prototxt"
weightsFile = "model/pose_iter_102000.caffemodel"
nPoints = 22
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


def hand_key_detect(image_path, frame=False):
    if type(frame) == bool:
        frame = cv2.imread(image_path)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight

    threshold = 0.1

    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    points = []

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)
    if points[0][1] < points[12][1] or points[0][1] < points[20][1] or points[0][1] < points[4][1]:
        frame = np.rot90(frame)
        hand_key_detect(0, frame=frame)
    else:
        # down points[0]
        # up points[12]
        # right points[20]
        # left points[4]
        # cv2.imshow('Output-Skeleton', frame[points[12][1]:points[0][1], points[4][0]-25:points[20][0]+30])
        # cv2.waitKey(0)
        return frame[points[12][1]:points[0][1], points[4][0]-25:points[20][0]+30]
