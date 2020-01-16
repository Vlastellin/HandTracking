import cv2
import run
import numpy as np
import argparse


def main(file):
    vidcap = cv2.VideoCapture(file)

    frames = list()
    # image = [imageio.imread('test.jpg')]
    # images = run.get_pic(image)
    success, image = vidcap.read()

    while success:
        image = np.rot90(image, k=3)
        frames.append(image)
        success, image = vidcap.read()

    print(len(frames))

    images = run.get_pic(frames)

    video = cv2.VideoWriter('output.mp4', 0x00000021, 15.0, (640, 480))

    for item in range(len(images)):
        video.write(cv2.imread("imgs/{}.png".format(str(item))))

    cv2.destroyAllWindows()
    video.release()


ap = argparse.ArgumentParser()
ap.add_argument('--video')
opts = ap.parse_args()
main(opts.video)
