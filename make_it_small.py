import cv2
import numpy as np


def small(image):
    image = cv2.resize(image, (140, 180))

    lt = image.tolist()

    for i in range(70):
        lt.insert(i, [[255, 255, 255] for x in range(240)])
        lt.append([[255, 255, 255] for y in range(240)])
    for i in range(len(lt)):
        if len(lt[i]) != 240:
            for z in range(50):
                lt[i].insert(z, [255, 255, 255])
                lt[i].append([255, 255, 255])

    lt = np.array(lt)
    lt = lt.astype(np.uint8)

    return lt
