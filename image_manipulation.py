from __future__ import division

import cv2
import numpy as np
import math


def crop_image(x, y, img, angle):
    h = 300
    w = 300

    d = int((h * h + w * w) ** (1 / 2))
    x = max(0, x - int(d / 2))
    y = max(0, y - int(d / 2))
    img = img[y:y + d, x:x + d]
    img = rotate_bound(img, angle)

    l = img.shape[0]
    img = img[int((l - h) / 2):int((l - h) / 2) + h, int((l - w) / 2):int((l - w) / 2) + w]

    return img


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def save_image(img, path):
    cv2.imwrite(path, img)


def convert_crop_points(detected_points):
    counter = 0
    point_one = []
    point_two = []

    for i in detected_points:
        if counter % 2 == 0:
            point_one.append(detected_points[counter])
        else:
            point_two.append(detected_points[counter])
        counter = counter + 1

    return point_one, point_two


def crop_points_cord(point_one, point_two, fingers):
    crop_point_one = []
    crop_point_two = []
    crop_angle = []

    # For four fingers
    for i in range(fingers):
        # Coordinates of the second detected point
        x1_cord = point_two[i][0]
        y1_cord = point_two[i][1]

        # Coordinates of the first detected point
        x2_cord = point_one[i][0]
        y2_cord = point_one[i][1]

        hypotenuse = y2_cord - y1_cord

        b_length = x1_cord - x2_cord

        alpha = b_length / hypotenuse
        alpha_angle = math.atan(alpha)

        crop_point_one.append(x1_cord)
        crop_point_two.append(y1_cord)
        crop_angle.append(alpha_angle)

    return crop_point_one, crop_point_two, crop_angle
