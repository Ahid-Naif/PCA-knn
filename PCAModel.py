import cv2
import numpy as np

"""
this class takes the training data & testing data 
then, convert these data into PCA space
"""
class PCAModel():
    # the constructor of the class
    # it runs once we make an instance of the class
    def __init__(self, trainingData, testingData, imageShape):
        self.__trainingDataMatrix = trainingData
        self.__testingDataMatrix = testingData
        self.__imageShape = imageShape

        self.__eigenVectors = []
        self.__mean = []
        self.__trainingPCAData = []
        self.__testingPCAData = []
        
        self.__computeEigenVectors()
        self.__convertData2PCA(self.__trainingDataMatrix)
        self.__convertData2PCA(self.__testingDataMatrix, isTesting=True)
    
    def __computeEigenVectors(self):
        print("Computing PCA ", end="... ")
        mean, eigenvectors = cv2.PCACompute(self.__trainingDataMatrix, mean=None)
        print("DONE")
        self.__mean = mean
        self.__eigenVectors = eigenvectors
        self.__eigenVectors = np.transpose(self.__eigenVectors)

    def __convertData2PCA(self, dataMatrix, isTesting=False):
        # this function converts data matrix into PCA space

        if isTesting:
            dataMatrix - self.__mean
        
        if isTesting:
            # if data matrix is for testing, store PCA data into testingPCA[] array
            self.__testingPCAData = np.matmul(dataMatrix, self.__eigenVectors)
        else:
            # if data matrix is for training, store PCA data into trainingPCA[] array
            self.__trainingPCAData = np.matmul(dataMatrix, self.__eigenVectors)

    def getPCA_Data(self):
        # return both training & testing data in PCA space

        return self.__trainingPCAData, self.__testingPCAData