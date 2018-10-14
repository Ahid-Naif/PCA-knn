import numpy as np
import cv2
from imutils import paths
import os

"""
This class takes tha path of a dataset
Then, generates the labels & data matrix that consists of flat images
"""
class PrepareModel():
    # the constructor of the class
    # it runs once we make an instance of the class
    def __init__(self, datasetPath, windowSize):
        self.__datasetPath = datasetPath
        self.__width = windowSize[0]
        self.__height = windowSize[1]

        self.__images = []
        self.__imageShape = []
        self.__imagesLabels = []
        self.__DataMatrix = np.zeros((2,2), dtype=np.float32) # just an initial value
        self.__numImages = 0

        self.__readImagesANDLabels()
        self.__createFlatDataMatrix()

    def __readImagesANDLabels(self):
        # this function is to read images & labels

        print("[INFO] Loading Images & Labels form " + str(self.__datasetPath), end="... ")
        imagesPaths = sorted(list(paths.list_images(self.__datasetPath)))
        for imagePath in imagesPaths:
            image = cv2.imread(imagePath) # read image
            label = imagePath.split(os.path.sep)[-3] # extract label
            if image is not None:
                image = cv2.resize(image, (self.__width, self.__height))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = np.float32(gray) / 255.0
                    
                self.__images.append(gray) # add image to images[] array
                self.__imagesLabels.append(label) # add label to imagesLabels[] array
                flippedImage = cv2.flip(gray, 1)
                self.__images.append(flippedImage)
                self.__imagesLabels.append(label) # add label to imagesLabels[] array
        print("DONE")

        self.__numImages = len(self.__images)
        print(str(self.__numImages // 2) + " images were loaded!")

    def __createFlatDataMatrix(self):
        # this function is to create the data matrix that consists of flat images

        print("[INFO] Creating Data Matrix", end="... ")
        self.__imageShape = self.__images[0].shape
        flatMatrixWidth = self.__numImages
        flatMatrixHeight = self.__imageShape[0] * self.__imageShape[1]
        self.__DataMatrix = np.zeros((flatMatrixWidth, flatMatrixHeight), dtype=np.float32)
        for i in range(0, self.__numImages):
            imageVector = self.__images[i].flatten()    
            self.__DataMatrix[i, :] = imageVector # fill in the data matrix
        
        print("DONE")
            
    def getDataANDLabels(self):
        # return data matrix along with the labels

        return self.__DataMatrix, self.__imagesLabels