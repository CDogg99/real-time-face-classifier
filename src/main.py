#!/usr/bin/env python2
import cv2
import numpy as np
import os
import argparse
import time
import math
import openface
import pickle

import align
import classifier
import config
import rep
from image import Image

fileDir = os.path.dirname(os.path.realpath(__file__))
model = os.path.join(fileDir, "..", "models", "nn4.small2.v1.t7")
net = openface.TorchNeuralNet(model = model, imgDim = config.ALIGNED_IMG_SIZE, cuda = True)

def main(args):
    cap = None
    if(args.video is not None):
        video = os.path.join(fileDir, "..", args.video)
        if(not os.path.isfile(video)):
            print("Video file not found")
            quit()
        cap = cv2.VideoCapture(video)
    else:
        cap = cv2.VideoCapture(0)
        cap.set(3, config.CAMERA_WIDTH)
        cap.set(4, config.CAMERA_HEIGHT)

    with open(args.classifier, 'rb') as f:
        (le, clf) = pickle.load(f)

    startTime = time.time()
    numFrames = 0
    numFaces = 0
    faceLocations = []
    while(cap.isOpened() or args.video is not None):
        ret, frame = cap.read()
        if(not ret):
            break
        numFrames += 1
        
        toWait = int(math.ceil(1.0/config.TARGET_FRAMERATE*1000))
        if(numFrames % config.PROCESS_NTH_FRAME == 0):
            preparedFrame = align.resizeImage(frame)
            faceLocations = align.getFaceLocationsArray([preparedFrame])[0]
            img = Image("", str(numFrames), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            alignedFaces = align.alignImages([img], [faceLocations])[0]
            reps = rep.getReps(alignedFaces, faceLocations, net)
            predictions = classifier.infer(reps, le, clf)
            for (person, confidence) in predictions:
                print("Guessed " + person + " with " + str(confidence) + " confidence")
            toWait = 1
        for (top, right, bottom, left) in faceLocations:
            numFaces += 1
            top *= int(config.DOWNSCALE)
            right *= int(config.DOWNSCALE)
            bottom *= int(config.DOWNSCALE)
            left *= int(config.DOWNSCALE)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.imshow("video", frame)
        if(cv2.waitKey(toWait) & 0xFF == ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(str(numFaces) + " faces detected in " + str(numFrames) + " frames")
    print("Face detection accuracy: " + str(float(numFaces)/(numFrames + 1) * 100.0))
    print("Average framerate: " + str(numFrames/(time.time() - startTime)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type = str, help = "Path to video")
    parser.add_argument("classifier", type = str, help = "Path to classifier")
    args = parser.parse_args()
    main(args)