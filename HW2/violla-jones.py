from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from math import floor
import numpy as np
import pickle
import scipy
import cv2
import os

xAxis = 0
yAxis = 1

#Paths to train images
path_to_train_images_face_folder = './images/face.train/train/face/'
path_to_train_images_non_face_folder = './images/face.train/train/non-face/'

#Paths to test images
path_to_test_images_face_folder = './images/face.test/test/face/'
path_to_test_images_non_face_folder = './images/face.test/test/non-face/'

path_to_test_image = './images/lena_color_512.tif'

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def integralImage(image):
    integralImage = np.zeros(image.shape)

    for xPx in range(image.shape[xAxis]):
        sum = 0
        for yPx in range(image.shape[yAxis]):
            sum+= image[xPx][yPx]
            if (xPx == 0):
                integralImage[xPx][yPx] = sum
            else:
                integralImage[xPx][yPx] = integralImage[xPx-1][yPx] + sum
    return integralImage

def showImage(label='Image', image=None):
    if image != None:
        cv2.imshow(label, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def getIntegralImages(images):
    iis = []
    for image in images:
       iis.append(integralImage(image)) 
    return iis

#Loading all images and push them into list
trainImagesFace = np.uint8(load_images_from_folder(path_to_train_images_face_folder))
trainImagesNonFace = np.uint8(load_images_from_folder(path_to_train_images_non_face_folder))

testImagesFace = np.uint8(load_images_from_folder(path_to_test_images_face_folder))
testImagesNonFace = np.uint8(load_images_from_folder(path_to_test_images_non_face_folder))

def haar_features(image):
    
    #image = np.array([[1,1,1,2],[1,2,1,1],[1,0,1,1],[0,0,2,2]])
    
    ii = integralImage(image)
    
    featuresA = []
    featuresB = []
    featuresC = []
    featuresD = []
    featuresE = []
    
    
    
    imageSizeX = image.shape[xAxis]
    imageSizeY = image.shape[yAxis]
    
    
    for filterHeight in range(0, imageSizeY):
        for filterWidth in range(1, imageSizeX, 2):
            for rectHeight in range(0, imageSizeY-filterHeight):
                for rectWidth in range(0, imageSizeX- filterWidth):
  
                    if (rectHeight == 0 and rectWidth == 0):
                        A = ii[filterHeight][floor(filterWidth/2)]
                        Summa = ii[filterHeight][filterWidth]
                        featuresA.append(2*A - Summa)
                    elif (rectHeight == 0):
                        Summa = ii[filterHeight][rectWidth + filterWidth] + ii[filterHeight][rectWidth-1]
                        A = ii[filterHeight][rectWidth + floor((filterWidth)/2)]
                        featuresA.append(2*A - Summa)
                    elif rectWidth == 0:
                        X = ii[rectHeight+filterHeight][floor((filterWidth)/2)]-ii[rectHeight-1][floor((filterWidth)/2)]
                        Y = ii[rectHeight+filterHeight][filterWidth]-ii[rectHeight-1][filterWidth] - X
                        featuresA.append(X - Y)
                    else:
                        A = ii[rectHeight-1][rectWidth-1]
                        B = ii[rectHeight-1][rectWidth+filterWidth]
                        C = ii[rectHeight + filterHeight][rectWidth-1]
                        D = ii[rectHeight+filterHeight][rectWidth+filterWidth]
                        
                        Summa = (A + D - B - C)
                        
                        B = ii[rectHeight-1][rectWidth+floor((filterWidth)/2)]
                        D = ii[rectHeight+filterHeight][rectWidth+floor((filterWidth)/2)]
                        
                        White = A + D - B - C
                        
                        featuresA.append(2*White - Summa)
        
        
                        
        
    for filterHeight in range(0, imageSizeY):
        for filterWidth in range(2, imageSizeX, 3):
            for rectHeight in range(0, imageSizeY-filterHeight):
                for rectWidth in range(0, imageSizeX- filterWidth):
  
                    if (rectHeight == 0 and rectWidth == 0):
                        Summa = ii[filterHeight][filterWidth]
                        LeftWhite = ii[filterHeight][floor(1/3*filterWidth)]                        
                        DarkUnionLeftWhite = ii[filterHeight][floor(2/3*filterWidth)]
                        C = Summa - DarkUnionLeftWhite
                        featuresB.append(2*(C + LeftWhite)-Summa)
                    elif (rectHeight == 0):
                        Summa = ii[filterHeight][rectWidth + filterWidth] - ii[filterHeight][rectWidth-1]
                        LeftWhite = ii[filterHeight][rectWidth + floor(1/3*filterWidth)]-ii[filterHeight][rectWidth-1]
                        RightWhite = ii[filterHeight][rectWidth+filterWidth]-ii[filterHeight][rectWidth + floor(2/3*filterWidth)]
                        featuresB.append(2*(LeftWhite + RightWhite)-Summa)
                    elif rectWidth == 0:
                        Summa = ii[rectHeight + filterHeight][filterWidth] - ii[rectHeight-1][filterWidth]
                        LeftWhite = ii[rectHeight + filterHeight][floor(1/3*filterWidth)]-ii[rectHeight-1][floor(1/3*filterWidth)]
                        BlackUnionLeftWhite = ii[rectHeight + filterHeight][floor(2/3*filterWidth)] - ii[rectHeight-1][floor(2/3*filterWidth)]
                        Black = BlackUnionLeftWhite - LeftWhite
                        RightWhite = Summa - BlackUnionLeftWhite
                        featuresB.append(LeftWhite + RightWhite - Black)
                        
                    else:
                        Summa = ii[rectHeight+filterHeight][rectWidth+filterWidth]+ii[rectHeight-1][rectWidth-1]-ii[rectHeight-1][rectWidth+filterWidth]-ii[rectHeight+filterHeight][rectWidth-1]
                        LeftWhite = ii[rectHeight+filterHeight][rectWidth+floor(1/3*filterWidth)]+ii[rectHeight-1][rectWidth-1]-ii[rectHeight-1][rectWidth+floor(1/3*filterWidth)]-ii[rectHeight+filterHeight][rectWidth-1]
                        BlackUnionLeftWhite = ii[rectHeight+filterHeight][rectWidth+floor(2/3*filterWidth)]+ii[rectHeight-1][rectWidth-1]-ii[rectHeight-1][rectWidth+floor(2/3*filterWidth)]-ii[rectHeight+filterHeight][rectWidth-1]
                        Black = BlackUnionLeftWhite - LeftWhite
                        RightWhite = Summa - BlackUnionLeftWhite
                        featuresB.append(LeftWhite + RightWhite - Black)
                        
                        
                        
    for filterHeight in range(1, imageSizeY, 2):
        for filterWidth in range(0, imageSizeX):
            for rectHeight in range(0, imageSizeY-filterHeight):
                for rectWidth in range(0, imageSizeX- filterWidth):
  
                    if (rectHeight == 0 and rectWidth == 0):
                        A = ii[floor(filterHeight/2)][filterWidth]
                        Summa = ii[filterHeight][filterWidth]
                        featuresC.append(2*A - Summa)
                    elif (rectHeight == 0):
                        Summa = ii[filterHeight][rectWidth + filterWidth] - ii[filterHeight][rectWidth-1]
                        A = ii[floor(filterHeight/2)][rectWidth + filterWidth]-ii[floor(filterHeight/2)][rectWidth-1]
                        featuresC.append(2*A - Summa)
                    elif rectWidth == 0:
                        Summa = ii[rectHeight+filterHeight][filterWidth]-ii[rectHeight-1][filterWidth]
                        Y = ii[rectHeight+floor(filterHeight/2)][filterWidth]-ii[rectHeight-1][filterWidth]
                        featuresC.append(2*Y-Summa)
                    else:
                        A = ii[rectHeight-1][rectWidth-1]
                        B = ii[rectHeight-1][rectWidth+filterWidth]
                        C = ii[rectHeight + filterHeight][rectWidth-1]
                        D = ii[rectHeight+filterHeight][rectWidth+filterWidth]
                        
                        Summa = (A + D - B - C)
                        
                        C = ii[rectHeight + floor(filterHeight/2)][rectWidth-1]
                        D = ii[rectHeight+floor(filterHeight/2)][rectWidth+filterWidth]
                        
                        White = A + D - B - C
                        
                        featuresC.append(2*White - Summa)
    #featuresD
    
    for filterHeight in range(2, imageSizeY, 3):
        for filterWidth in range(0, imageSizeX):
            for rectHeight in range(0, imageSizeY-filterHeight):
                for rectWidth in range(0, imageSizeX- filterWidth):
  
                    if (rectHeight == 0 and rectWidth == 0):                        
                        Summa = ii[filterHeight][filterWidth]
                        downWhite = ii[floor(1/3*filterHeight)][filterWidth]
                        blackUnionDownWhite = ii[floor(2/3*filterHeight)][filterWidth]
                        black = blackUnionDownWhite - downWhite
                        upWhite = Summa - blackUnionDownWhite
                        featuresD.append(downWhite+upWhite-black)
                    elif (rectHeight == 0):
                        Summa = ii[filterHeight][rectWidth + filterWidth] - ii[filterHeight][rectWidth-1]
                        downWhite = ii[floor(1/3*filterHeight)][rectWidth+filterWidth]-ii[floor(1/3*filterHeight)][rectWidth-1]
                        blackUnionDownWhite = ii[floor(2/3*filterHeight)][rectWidth+filterWidth]-ii[floor(2/3*filterHeight)][rectWidth-1]
                        black = blackUnionDownWhite - downWhite
                        upWhite = Summa - blackUnionDownWhite
                        featuresD.append(downWhite+upWhite-black)
                    elif rectWidth == 0:
                        
                        Summa = ii[rectHeight+filterHeight][filterWidth]-ii[rectHeight-1][filterWidth]
                        downWhite = ii[rectHeight + floor(1/3*filterHeight)][filterWidth] - ii[rectHeight-1][filterWidth]
                        
                        blackUnionDownWhite = ii[rectHeight + floor(2/3*filterHeight)][filterWidth] - ii[rectHeight-1][filterWidth]
                        black = blackUnionDownWhite - downWhite
                        upWhite = Summa - blackUnionDownWhite
                        featuresD.append(downWhite+upWhite-black)
                    else:
                        summa = ii[rectHeight+filterHeight][rectWidth + filterWidth]-ii[rectHeight-1][rectWidth+filterWidth]-ii[rectHeight+filterHeight][rectWidth-1]+ii[rectHeight-1][rectWidth-1]
                        downWhite = ii[rectHeight + floor(1/3*filterHeight)][rectWidth+filterWidth] + ii[rectHeight-1][rectWidth-1]-ii[rectHeight-1][rectWidth+filterWidth]-ii[rectHeight+floor(1/3*filterHeight)][rectWidth-1]
                        blackUnionDownWhite = ii[rectHeight + floor(2/3*filterHeight)][rectWidth+filterWidth] + ii[rectHeight-1][rectWidth-1]-ii[rectHeight-1][rectWidth+filterWidth]-ii[rectHeight+floor(2/3*filterHeight)][rectWidth-1]
                        black = blackUnionDownWhite - downWhite
                        upWhite = summa - blackUnionDownWhite
                        featuresD.append(downWhite+upWhite-black)
              
    #featuresE
                
    for filterSize in range(1, imageSizeY, 2):
            for rectHeight in range(0, imageSizeY-filterSize):
                for rectWidth in range(0, imageSizeX- filterSize):
                    if (rectHeight == 0 and rectWidth == 0):
                        summa = ii[filterSize][filterSize]
                        blackDown = ii[floor(1/2*filterSize)][floor(1/2*filterSize)]
                        blackUp = ii[filterSize][filterSize] + blackDown - ii[filterSize][filterSize-floor(1/2*filterSize)-1] - ii[filterSize-floor(1/2*filterSize)-1][filterSize]
                        featuresE.append(summa - 2*(blackDown + blackUp))
                    elif (rectHeight == 0):
                        summa = ii[filterSize][rectWidth + filterSize] - ii[filterSize][rectWidth-1]                        
                        blackDown = ii[floor(1/2*filterSize)][rectWidth+floor(1/2*filterSize)] - ii[floor(1/2*filterSize)][rectWidth-1]                         
                        blackUp = ii[filterSize][rectWidth + filterSize] + ii[floor(1/2*filterSize)][rectWidth + floor(1/2*filterSize)] - ii[filterSize][rectWidth+floor(1/2*filterSize)] - ii[floor(1/2*filterSize)][rectWidth+floor(1/2*filterSize)+1]
                        featuresE.append(summa - 2*(blackDown + blackUp))                        
                    elif rectWidth == 0:
                        summa = ii[rectHeight + filterSize][filterSize] - ii[rectHeight-1][filterSize]                        
                        blackDown = ii[rectHeight + floor(1/2*filterSize)][floor(1/2*filterSize)] - ii[rectHeight-1][floor(1/2*filterSize)]                         
                        blackUp = ii[rectHeight + filterSize][filterSize] + ii[rectHeight+ floor(1/2*filterSize)][floor(1/2*filterSize)] - ii[rectHeight + floor(1/2*filterSize)][filterSize] - ii[rectHeight + filterSize][floor(1/2*filterSize)]
                        featuresE.append(summa - 2*(blackDown + blackUp))
                    else:
                        summa = ii[rectHeight+filterSize][rectWidth + filterSize]-ii[rectHeight-1][rectWidth+filterSize]-ii[rectHeight+filterSize][rectWidth-1]+ii[rectHeight-1][rectWidth-1]                        
                        blackDown = ii[rectHeight + floor(1/2*filterSize)][rectWidth + floor(1/2*filterSize)] + ii[rectHeight-1][rectWidth-1] - ii[rectHeight-1][rectWidth+floor(1/2*filterSize)] - ii[rectHeight+floor(1/2*filterSize)][rectWidth-1]                        
                        blackUp = ii[rectHeight+filterSize][rectWidth + filterSize] + ii[rectHeight+ floor(1/2*filterSize)][rectWidth+floor(1/2*filterSize)] - ii[rectHeight +floor(1/2*filterSize)][rectWidth + floor(1/2*filterSize)+1] - ii[rectHeight + filterSize][rectWidth + floor(1/2*filterSize)]
                        featuresE.append(summa - 2*(blackDown + blackUp))

                    
    res = np.hstack((featuresA, featuresB, featuresC, featuresD, featuresE))
    res = np.uint8(res)
    return res


def trainStrongClassificatorWithSerialization():

    X = []
    Y = []

    
    weights = []

    #init with first image face
    X.append(haar_features(trainImagesFace[0]))
    Y.append(1)
    weights.append(1/(2*2429))

    weights = np.array(weights)

    for imageIndex in range(1,trainImagesFace.__len__()):
        X = np.vstack((X, haar_features(trainImagesFace[imageIndex])))
        Y.append(1)
        weights.append(1/(2*2429))

    for imageIndex in range(0,trainImagesNonFace.__len__()):
        X = np.vstack((X, haar_features(trainImagesNonFace[imageIndex])))
        Y.append(-1)
        weights.append(1/(2*4548))

    Y = np.array(Y)
    Y.transpose()

    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),algorithm="SAMME.R",n_estimators=30)
    model.fit(X, Y, weights)

    output = open('data.pkl', 'wb')
    pickle.dump(model, output, 2)
    output.close()

def getStrongClassificatorFromTheFile():
    file = open("data.pkl",'rb')
    model = pickle.load(file)
    file.close()
    return model

def checkForFacesAndNonFacesInTestSet(strongClassif, testImagesFace, testImagesNonFace ):
    
    countLose = 0
    countWin = 0
    
    '''for face in range(0,testImagesFace.__len__()):
        imageClass = strongClassif.predict(haar_features(testImagesFace[face]).reshape((1, -1)))
        if imageClass[0] == 1:
            countWin = countWin + 1
        else:
            countLose = countLose + 1
        print("Number of loosed faces: ", countLose)
        print("Number of finded faces: ", countWin)'''
    
    
    for face in range(0,testImagesNonFace.__len__()):
        imageClass = strongClassif.predict(haar_features(testImagesNonFace[face]).reshape((1, -1)))
        if imageClass[0] == -1:
            countWin = countWin + 1
        else:
            countLose = countLose + 1
        print("Number of fail predict: ", countLose)
        print("Number of correct predict: ", countWin)
    
    

def findAndDrawFace(strongClassif, image, startSizeX, startSizeY): 
    
    small = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    
    fig,ax = plt.subplots(1)
    ax.imshow(small)
    rect = patches.Rectangle((500,100),40,30,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()
    
strongClassif = getStrongClassificatorFromTheFile()  
checkForFacesAndNonFacesInTestSet(strongClassif, testImagesFace,testImagesNonFace)
#trainStrongClassificatorWithSerialization()
#img = cv2.imread(path_to_test_image,cv2.IMREAD_GRAYSCALE)
#haar_features(None)
#drawFace(None, img)
