#!/usr/bin/env python2
import face_recognition
import face_recognition_models
import dlib
import cv2
import numpy as np
import errno
import os
import time

import constants
from image import Image

# Landmark detection
PREDICTOR_68_POINT_MODEL = face_recognition_models.pose_predictor_model_location()
POSE_PREDICTOR_68_POINT = dlib.shape_predictor(PREDICTOR_68_POINT_MODEL)

"""
Get all landmarks for each face in image
"""
def getLandmarks(image, faceLocations):
    # Each element stores an array indicating landmarks for a face
    landmarksArray = []
    for (top, right, bottom, left) in faceLocations:
        # Upscale values to compensate for earlier downscaling
        top *= constants.DOWNSCALE
        right *= constants.DOWNSCALE
        bottom *= constants.DOWNSCALE
        left *= constants.DOWNSCALE
        rect = dlib.rectangle(left, top, right, bottom)
        landmarksArray.append(POSE_PREDICTOR_68_POINT(image, rect))
    # Same as landmarks variable, with the actual data stored as tuples rather than points
    landmarksAsTuples =[[(p.x, p.y) for p in landmarks.parts()] for landmarks in landmarksArray]
    return landmarksAsTuples

"""
Read in image as BRG and change color space to RGB
"""
def getImage(imagePath):
    bgrImage = cv2.imread(imagePath, 1)
    if(bgrImage is None):
        return None
    rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
    return rgbImage

"""
Resize the given image matrix by a factor of 1/constants.DOWNSCALE
"""
def resizeImage(image):
    resizedImage = cv2.resize(image, (0, 0), fx=(1.0/constants.DOWNSCALE), fy=(1.0/constants.DOWNSCALE))
    return resizedImage

"""
Make the given directory if it doesn't exist
From OpenFace
"""
def mkdirP(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

"""
Write the given image to outDir
"""
def writeImage(image, outDir):
    outDir = os.path.join(outDir, image.clss)
    mkdirP(outDir)
    outDir = os.path.join(outDir, os.path.splitext(image.name)[0] + ".png")
    cv2.imwrite(outDir, cv2.cvtColor(image.imgMat, cv2.COLOR_RGB2BGR))

"""
Returns an array of images with aligned faces
images - an array of image matrices
faceLocationsArray - an array of face locations within each image
"""
def alignImages(images, faceLocationsArray):
    alignedImages = []
    # Loop through each image and its face locations
    for faceLocations, image in zip(faceLocationsArray, images):
        landmarksArray = getLandmarks(image.imgMat, faceLocations)
        # Loop through the array of landmarks for faces in this image
        curFace = 0
        for landmarks in landmarksArray:
            npLandmarks = np.float32(landmarks)
            npLandmarkIndices = np.array(constants.OUTER_EYES_AND_NOSE)

            H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                constants.ALIGNED_IMG_SIZE * constants.MINMAX_TEMPLATE[npLandmarkIndices])
            thumbnail = cv2.warpAffine(image.imgMat, H, (constants.ALIGNED_IMG_SIZE,
                constants.ALIGNED_IMG_SIZE))

            name = image.name
            if(len(landmarksArray) > 1):
                name += str(curFace)

            alignedImages.append(Image(image.clss, name, thumbnail))
            curFace += 1
    return alignedImages

def main():
    fileDir = os.path.dirname(os.path.realpath(__file__))
    imgDir = os.path.join(fileDir, "..", "images")
    rawDir = os.path.join(imgDir, "raw")
    outputDir = os.path.join(imgDir, "aligned")
    mkdirP(outputDir)

    dirStack = []
    dirStack.append(rawDir)

    images = []
    resizedImages = []
    numImages = 0

    startTime = time.time()

    while(len(dirStack)):
        parentDir = dirStack.pop()
        entries = os.listdir(parentDir)
        clss = os.path.basename(parentDir)
        print(clss)
        for entry in entries:
            path = os.path.join(parentDir, entry)
            if(os.path.isdir(path)):
                dirStack.append(path)
            else:
                numImages += 1
                img = getImage(path)
                images.append(Image(clss, entry, img))
                resizedImages.append(resizeImage(img))
                if(len(images) == constants.BATCH_SIZE):
                    faceLocationsArray = face_recognition.batch_face_locations(resizedImages, constants.UPSAMPLE, constants.BATCH_SIZE)
                    for image in alignImages(images, faceLocationsArray):
                        writeImage(image, outputDir)
                    images = []
                    resizedImages = []
        # Finish aligning remaining images for this class
        if(len(images) > 0):
            faceLocationsArray = face_recognition.batch_face_locations(resizedImages, constants.UPSAMPLE, len(images))
            for image in alignImages(images, faceLocationsArray):
                writeImage(image, outputDir)
            images = []
            resizedImages = []

    cv2.destroyAllWindows()
    print("Time to align images: " + str(time.time() - startTime))
    print("Total images processed: " + str(numImages))

if __name__ == '__main__':
    main()
