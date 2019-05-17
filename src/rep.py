
import config

def getReps(alignedFaces, faceLocations, net):
    reps = []
    i = 0
    for (top, right, bottom, left) in faceLocations:
        right *= config.DOWNSCALE
        left *= config.DOWNSCALE
        alignedFace = alignedFaces[i]
        rep = net.forward(alignedFace.imgMat)
        reps.append((int((right + left)/2.0), rep))
        i += 1
    reps = sorted(reps, key=lambda x: x[0])
    return reps

def getRepString(image, net):
    if(image is None):
        raise Exception("No image given")
    rep = net.forward(image)
    output = ""
    i = 0
    while(i < len(rep) - 1):
        output += str(rep[i]) + ","
        i += 1
    output += str(rep[i])
    return output
    