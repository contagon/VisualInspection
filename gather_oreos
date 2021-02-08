import cv2
import os
import argparse
from pathlib import Path

def get_count(capture):
    # Try to get most last image in folder, so we don't overwrite images
    try:
        last = sorted(os.listdir(f'data/{capture}'))[-1]
        count = int(last[:-4])+1
    # Otherwise set count to 
    except:
        count = 0

    return count


def main():
    capture = 'good'
    record  = False
    cv2.namedWindow("saving")

    # make all folders
    Path(f'data/good').mkdir(parents=True, exist_ok=True)
    Path(f'data/bad').mkdir(parents=True, exist_ok=True)
    Path(f'data/ugly').mkdir(parents=True, exist_ok=True)
    count = get_count(capture)
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args = vars(ap.parse_args())

    cap = cv2.VideoCapture(1)
    
    ret, first_img = cap.read()
    
    gray2 = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    while True:
        ret, img = cap.read()
        oreo = img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        frameDelta = cv2.absdiff(gray2, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # loop over the contours
        for c in cnts[0]:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                  continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            l = max(h,w)
            oreo = img[y:(y + l),x:(x + l)]

        # Save Image
        cv2.imshow('saving', oreo)
        if record:
            cv2.imwrite(f'data/{capture}/{count:06d}.jpg', oreo)
            count += 1

        # Process key presses
        key = cv2.waitKey(100)
        if key == ord('q'):
            break

        if key == ord('g'):
            capture = 'good'
            count = get_count(capture)
            print("Switching to good folder")

        if key == ord('b'):
            capture = 'bad'
            count = get_count(capture)
            print("Switching to bad folder")

        if key == ord('u'):
            capture = 'ugly'
            count = get_count(capture)
            print("Switching to ugly folder")

        if key == ord('r'):
            record = True
            print("Starting to save")

        if key == ord('s'):
            record = False
            print("Stopping saving")

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
