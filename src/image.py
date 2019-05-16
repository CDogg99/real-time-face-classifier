
"""
Object containing image metadata.
From OpenFace
"""
class Image:

    """
    Constructor for Image class
    clss - class for image
    name - name of image
    imgMat - actual image matrix (RGB)
    """
    def __init__(self, clss, name, imgMat):
        self.clss = clss
        self.name = name
        self.imgMat = imgMat

    """String representation for printing."""
    def __repr__(self):
        return "({}, {})".format(self.clss, self.name)
