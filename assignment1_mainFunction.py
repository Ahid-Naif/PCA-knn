import cv2
import pickle
import os.path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from PrepareModel import PrepareModel # import a class that I have made
from PCAModel import PCAModel # import a class that I have made

imageDimensions = (100, 100) # to resize all images to this size
### define files
trainingDataFile = os.path.isfile("trainingData.p")
trainingLabelsFile = os.path.isfile("trainingLabels.p")
testingDataFile = os.path.isfile("testingData.p")
testingLabelsFile = os.path.isfile("testingLabels.p")
trainingPCAFile = os.path.isfile("trainingPCA.p")
testingPCAFile = os.path.isfile("testingPCA.p")
###
if trainingDataFile and trainingLabelsFile:
    trainingData = pickle.load(open("trainingData.p", "rb" ))
    trainingLabels = pickle.load(open("trainingLabels.p", "rb" ))
else:
    # make an instance for training by & pass the training dataset path, and image size
    training = PrepareModel("TrainingSet", imageDimensions)
    
    trainingData, trainingLabels = training.getDataANDLabels() # get training data & labels

    ###
    pickle.dump(trainingData, open("trainingData.p", "wb" ))
    pickle.dump(trainingLabels, open("trainingLabels.p", "wb"))
    ###

if testingDataFile and testingLabelsFile:
    testingData = pickle.load(open("testingData.p", "rb" ))
    testingLabels = pickle.load(open("testingLabels.p", "rb" ))
else:
    # make an instance for testing by & pass the testing dataset path, and image size
    testing = PrepareModel("TestingSet", imageDimensions)
    testingData, testingLabels = testing.getDataANDLabels() # get testing data & labels

    ###
    pickle.dump(testingData, open("testingData.p", "wb"))
    pickle.dump(testingLabels, open("testingLabels.p", "wb"))
    ###

if trainingPCAFile and testingPCAFile:
    trainingPCA = pickle.load(open("trainingPCA.p", "rb" ))
    testingPCA = pickle.load(open("testingPCA.p", "rb" ))
else:
    # make an instance of pca
    pca = PCAModel(trainingData, testingData, imageDimensions)

    trainingPCA, testingPCA = pca.getPCA_Data() # get training data & testing data in PCA space

    ###
    pickle.dump(trainingPCA, open("trainingPCA.p", "wb"))
    pickle.dump(testingPCA, open("testingPCA.p", "wb"))
    ###    

for k in range(1,30,2): # try many k values from 1 to 30 with a step of 2
    knnModel = KNeighborsClassifier(n_neighbors=k)
    knnModel.fit(trainingPCA, trainingLabels)
    predictions = knnModel.predict(testingPCA)

    # show a final classification report demonstrating the accuracy of the classifier
    print("EVALUATION ON TESTING DATA")
    print("Value of k: " + str(k))
    print(classification_report(testingLabels, predictions))