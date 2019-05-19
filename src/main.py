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
net = openface.TorchNeuralNet(model = model, imgDim = config.ALIGNED_IMG_SIZE, cuda = config.GPU_REPS)

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

    with open(os.path.join(args.featuresDir, "classifier.pkl"), 'rb') as f:
        (le, clf) = pickle.load(f)

    labelsPath = os.path.join(args.featuresDir, "labels.csv")
    repsPath = os.path.join(args.featuresDir, "reps.csv")

    startTime = time.time()
    numFrames = 0
    numFaces = 0
    faceLocations = []
    predictions = []
    remainingUnknown = config.RETRAIN_REQUIREMENT
    saveReps = False
    collectedReps = []
    while(cap.isOpened() or args.video is not None):
        ret, frame = cap.read()
        if(not ret):
            break
        numFrames += 1
        
        toWait = int(math.ceil(1.0/config.TARGET_FRAMERATE*1000))
        # Process image if on nth frame or if reps are being saved
        if(numFrames % config.PROCESS_NTH_FRAME == 0 or saveReps):
            preparedFrame = align.resizeImage(frame)
            faceLocations = align.getFaceLocationsArray([preparedFrame])[0]
            img = Image("", str(numFrames), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            alignedFaces = align.alignImages([img], [faceLocations])[0]
            reps = rep.getReps(alignedFaces, faceLocations, net)
            predictions = classifier.infer(reps, le, clf)
            # Only seek to train new face while only 1 unknown person in frame
            if(len(faceLocations) == 1 and predictions[0][0] == "unknown"):
                if(saveReps):
                    collectedReps.append(reps[0])
                else:
                    remainingUnknown -= 1
                    if(remainingUnknown == 0):
                        saveReps = True
                        remainingUnknown = config.RETRAIN_REQUIREMENT
            else:
                remainingUnknown = config.RETRAIN_REQUIREMENT
                saveReps = False
                collectedReps = []
            for (person, confidence) in predictions:
                print("Guessed " + person + " with " + str(confidence) + " confidence")
            numFaces += len(faceLocations)
            toWait = 1

        # Collected enough reps - write to files and retrain
        if(len(collectedReps) == config.NUM_REPS_TO_SAVE):
            labels = open(labelsPath, "a+")
            reps = open(repsPath, "a+")
            newClassName = "person-" + str(len(le.classes_))
            for repArray in collectedReps:
                labels.write(newClassName + "\n")
                reps.write(rep.getRepStringFromRep(repArray[1]) + "\n")
            labels.close()
            reps.close()
            classifier.train(labelsPath, repsPath, args.featuresDir)
            with open(os.path.join(args.featuresDir, "classifier.pkl"), 'rb') as f:
                (le, clf) = pickle.load(f)
            saveReps = False
            collectedReps = []

        # Draw boxes around faces with labels at bottom
        for (top, right, bottom, left), (person, confidence) in zip(faceLocations, predictions):
            top *= int(config.DOWNSCALE)
            right *= int(config.DOWNSCALE)
            bottom *= int(config.DOWNSCALE)
            left *= int(config.DOWNSCALE)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, person, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Show frame
        cv2.imshow("Video", frame)
        if(cv2.waitKey(toWait) & 0xFF == ord("q")):
            break

    # Clean-up
    cap.release()
    cv2.destroyAllWindows()

    # Print results
    print(str(numFaces) + " faces detected in " + str(numFrames) + " frames")
    print("Face detection accuracy: " + str(float(numFaces)/(numFrames + 1) * 100.0))
    print("Average framerate: " + str(numFrames/(time.time() - startTime)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type = str, help = "Path to video")
    parser.add_argument("featuresDir", type = str, help = "Path to features directory")
    args = parser.parse_args()
    main(args)