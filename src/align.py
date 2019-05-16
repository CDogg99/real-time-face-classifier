#!/usr/bin/env python2
import face_recognition
import face_recognition_models
import dlib
import cv2
import numpy as np
import errno
import os
import time

import config
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
        top *= int(config.DOWNSCALE)
        right *= int(config.DOWNSCALE)
        bottom *= int(config.DOWNSCALE)
        left *= int(config.DOWNSCALE)
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
Resize the given image matrix by a factor of 1/config.DOWNSCALE and
change to gray colorspace for performance
"""
def resizeImage(image):
    resizedImage = cv2.resize(image, (0, 0), fx=(1.0/config.DOWNSCALE), fy=(1.0/config.DOWNSCALE))
    return cv2.cvtColor(resizedImage, cv2.COLOR_RGB2GRAY)

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
            npLandmarkIndices = np.array(config.OUTER_EYES_AND_NOSE)

            H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                config.ALIGNED_IMG_SIZE * config.MINMAX_TEMPLATE[npLandmarkIndices])
            thumbnail = cv2.warpAffine(image.imgMat, H, (config.ALIGNED_IMG_SIZE,
                config.ALIGNED_IMG_SIZE))
            
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
    numFaces = 0

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
                faceLocationsArray = None
                if(config.GPU and len(images) == config.BATCH_SIZE):
                    # GPU batch processing
                    faceLocationsArray = face_recognition.batch_face_locations(resizedImages, config.UPSAMPLE, config.BATCH_SIZE)
                elif(not config.GPU):
                    faceLocationsArray = []
                    for resizedImage in resizedImages:
                        faceLocations = face_recognition.face_locations(resizedImage, config.UPSAMPLE)
                        faceLocationsArray.append(faceLocations)
                if(faceLocationsArray is not None):
                    for faceLocations in faceLocationsArray:
                        numFaces += len(faceLocations)
                    for image in alignImages(images, faceLocationsArray):
                        writeImage(image, outputDir)
                    images = []
                    resizedImages = []
        # Finish aligning remaining images for this class if GPU enabled
        if(config.GPU and len(images) > 0):
            # GPU batch processing
            faceLocationsArray = face_recognition.batch_face_locations(resizedImages, config.UPSAMPLE, len(images))
            for faceLocations in faceLocationsArray:
                numFaces += len(faceLocations)
            for image in alignImages(images, faceLocationsArray):
                writeImage(image, outputDir)
            images = []
            resizedImages = []

    finishTime = time.time()
    cv2.destroyAllWindows()
    print("Time to align images: " + str(finishTime - startTime))
    print("Total images processed: " + str(numImages))
    print("Total faces detected: " + str(numFaces))
    print("Images/second: " + str(numImages/(finishTime - startTime)))

if __name__ == '__main__':
    main()
