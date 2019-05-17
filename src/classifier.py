#!/usr/bin/env python2
import pickle
import os
import pandas as pd
import numpy as np
import argparse
import openface

from operator import itemgetter

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import align
import rep
import config
from image import Image

def train(labelsPath, repsPath, outDir):
    labels = []
    file = open(labelsPath, "r")
    for line in file:
        labels.append(line.rstrip("\n"))
    file.close()

    embeddings = pd.read_csv(repsPath, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)

    clf = SVC(C=1, kernel='linear', probability=True)

    clf.fit(embeddings, labelsNum)

    with open(os.path.join(outDir, "classifier.pkl"), 'w') as f:
        pickle.dump((le, clf), f)

def infer(reps, labelEncoder, classifier):
    results = []
    for rep in reps:
        repValues = rep[1].reshape(1, -1)
        predictions = classifier.predict_proba(repValues).ravel()
        maxI = np.argmax(predictions)
        person = labelEncoder.inverse_transform(maxI)
        confidence = predictions[maxI]
        results.append((person.decode("utf-8"), confidence))
    return results

def main(args):
    fileDir = os.path.dirname(os.path.realpath(__file__))
    model = os.path.join(fileDir, "..", "models", "nn4.small2.v1.t7")
    net = openface.TorchNeuralNet(model = model, imgDim = config.ALIGNED_IMG_SIZE, cuda = config.GPU_REPS)

    outputDir = os.path.join(fileDir, "..", "images", "other", "test", "aligned")

    if args.mode == "infer":
        with open(args.classifier, 'rb') as f:
            (le, clf) = pickle.load(f)
        print("Giving predictions from left to right in image")
        for imagePath in args.images:
            imageMat = align.getImage(imagePath)
            if(imageMat is None):
                raise Exception("No image at " + imagePath)
            resizedImageMat = align.resizeImage(imageMat)
            faceLocations = align.getFaceLocationsArray([resizedImageMat])[0]
            if(len(faceLocations) == 0):
                print("No faces detected")
                continue
            file = os.path.basename(imagePath)
            name = os.path.splitext(file)[0]
            image = Image("", name, imageMat)
            alignedFaces = align.alignImages([image], [faceLocations])[0]
            for face in alignedFaces:
                align.writeImage(face, outputDir)
            reps = rep.getReps(alignedFaces, faceLocations, net)
            predictions = infer(reps, le, clf)
            print(file)
            for (person, confidence) in predictions:
                print(person + " " + str(confidence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', help="Mode")

    inferParser = subparsers.add_parser('infer', help='Predict faces in the given images')
    inferParser.add_argument('classifier', type=str, help='Path to classifier')
    inferParser.add_argument('images', type=str, nargs='+', help="Input images")
    
    args = parser.parse_args()
    main(args)