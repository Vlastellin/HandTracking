from __future__ import division

import cv2
import numpy as np
import os
from imutils import paths
from image_manipulation import convert_crop_points, crop_points_cord, crop_image, save_image

IMAGE_FOLDER = "dataset/"
protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
cropped_images_folder = "cropped/"

nPoints = 22
POSE_PAIRS = [[5, 5], [6, 6], [9, 9], [10, 10], [13, 13], [14, 14], [17, 17], [18, 18]]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

fingers = 4
counter = 0
image_files = list(paths.list_images(IMAGE_FOLDER))

for image in range(len(image_files)):
    frame = cv2.imread(image_files[image])
    base = os.path.basename(image_files[image])
    print(os.path.splitext(base)[0])
    frameCopy = np.copy(frame)
    frame_phalanxes = np.copy(frame)
    image = cv2.cvtColor(frame_phalanxes, cv2.COLOR_BGR2GRAY)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth / frameHeight

    threshold = 0.1

    inHeight = 368
    inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    points = []

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255),
                        2, lineType=cv2.LINE_AA)

            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    detected_points = []
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB]:
            circle = cv2.circle(frame_phalanxes, (points[partA]), 8, (0, 255, 255), thickness=-1)
            detected_points.append(points[partA])

    point_one, point_two = convert_crop_points(detected_points)
    crop_point_one, crop_point_two, crop_angle = crop_points_cord(point_one, point_two, fingers)

    # for four fingers
    for i in range(fingers):
        print('Rotation angle', -np.rad2deg(crop_angle[i]))
        # crop image by vector angle
        crop = crop_image(crop_point_one[i], crop_point_two[i], frame_phalanxes, -np.rad2deg(crop_angle[i]))
        path = os.path.join(cropped_images_folder, "{}.jpg".format(str(i)))
        save_image(crop, path)
        cv2.imshow('Crop', crop)
        cv2.waitKey(0)

    print('------------------------------')
    cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    cv2.imwrite('phalanxes.jpg', frame_phalanxes)
    cv2.waitKey(0)
