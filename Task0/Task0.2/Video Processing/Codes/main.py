import cv2
import numpy as np
import os

def partA():
    os.chdir('../Videos')
    video = cv2.VideoCapture('RoseBloom.mp4')
    # getting the frame rate for given video.
    fps = video.get(5)
    # setting the frame to the start of 6th second.
    video.set(1, fps*5)
    ret, frame = video.read()
    os.chdir('../Generated')
    cv2.imwrite('frame_as_6.jpg', frame)

def partB():
    img = cv2.imread('frame_as_6.jpg', 1)
    # setting blue and green channel to 0.
    img[:, :, 0] = 0
    img[:, :, 1] = 0
    cv2.imwrite('frame_as_6_red.jpg', img)

partA()
partB()
