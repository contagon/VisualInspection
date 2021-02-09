import cv2
import os
import argparse

import torch
import torch.nn as nn
from opencv_transforms import transforms

from network import CNN
import numpy as np


def main():
    # load pytorch model
    c, h, w = 3, 64, 64
    model = CNN(c, h, w)
    model.load_state_dict( torch.load('model_best.pkl') )

    # load transforms needed
    transform = transforms.Compose([
                        transforms.Resize(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
    class_name = ['ugly', 'good', 'bad']
    class_color = [(255,0,0), (0,255,0), (0,0,255)]

    cv2.namedWindow("Result")
    cap = cv2.VideoCapture(2)
    ret, first_img = cap.read()

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file", default='ran.avi')
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args = vars(ap.parse_args())

    # get ready to save file
    height, width, _ = first_img.shape
    vout = cv2.VideoWriter(args['video'], cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))

    # get first image as our baseline
    gray2 = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    while True:
        ret, img = cap.read()
        oreo = img

        # Grascale and blur our iamge
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Take difference based on baseline, threshold
        frameDelta = cv2.absdiff(gray2, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Find contour of image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # loop over the contours
        for c in cnts[0]:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                  continue
            # compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)
            l = max(h,w)
            # get oreo and get ready for cnn
            oreo = img[y:(y + l),x:(x + l)]
            oreo = cv2.cvtColor(oreo, cv2.COLOR_BGR2RGB)
            oreo = transform( oreo ).unsqueeze(0)

            # run oreo through CNN
            result = model( oreo )
            result = result.argmax(1).item()

            # and update the text and rectangle
            cv2.rectangle(img, (x, y), (x+l, y+l), class_color[result], 2)
            cv2.putText(img, class_name[result], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, class_color[result], 2, )

        # Show Image
        cv2.imshow('Result', img)
        vout.write(img)

        # Process key presses
        key = cv2.waitKey(100)
        if key == ord('q'):
            break

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
