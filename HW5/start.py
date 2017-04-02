import scipy.ndimage as nd
import numpy as np
import cv2

path_to_first_image = "./images/monkey5_001.jpg"
path_to_second_image = "./images/monkey5_002.jpg"

path_to_result_image = "./output/lucasKanada.jpg"

firstImage = cv2.imread(path_to_first_image,cv2.IMREAD_GRAYSCALE)
secondImage = cv2.imread(path_to_second_image,cv2.IMREAD_GRAYSCALE)


def convoluteImage(image, imageFilter):
    
    image = np.int32(image)
    imageSizeX, imageSizeY = image.shape

    #make border
    imageZeros = np.zeros((imageSizeX+4, imageSizeY+4))
    
    for sizeX in range(0, imageSizeX):
        for sizeY in range(0, imageSizeY):
            imageZeros[sizeX+2][sizeY+2] = image[sizeX,sizeY]

    #print(imageZeros)
    #print(imageZeros[0:5,0:5])
    
    newOne = []
    
    for sizeX in range(0, imageSizeX):
        newOneStr = []
        for sizeY in range(0, imageSizeY):
            newOneStr.append(np.sum(np.dot(imageZeros[sizeX:5+sizeX,sizeY:5+sizeY],imageFilter)))
        newOne.append(newOneStr)

    newOne = np.float16(newOne)
    
    return newOne
    

def lucasKanada(firstGrayImage, secondGrayImage):
    
    #firstGrayImage = [[1,2,3,0,1],[1,2,3,2,1],[1,2,3,2,1],[1,2,3,2,1],[1,0,3,2,1]]
    
    #secondGrayImage = [[3,2,0,2,1],[1,2,0,2,1],[1,2,0,2,1],[1,2,0,2,1],[1,0,0,2,1]]
    #blur
    firstBlur = np.array([[1],[4],[6],[4],[1]])
    secondBlur = np.array([1,4,6,4,1])
    
    #
    firstApperture = np.array([[-1],[8],[0],[-8],[1]])
    secondApperture = np.array([1,8,0,-8,1])
    
    Ix = nd.sobel(firstGrayImage,1)
    Iy = nd.sobel(firstGrayImage,0)
    
    Ix2 = np.dot(Ix,Ix)
    Iy2 = np.dot(Iy,Iy)
    
    u = []
    v = []
    
    
    imageFilter = firstBlur*secondBlur
    
    imageApperture = 1/24 * (firstApperture*secondApperture)
    
    imageApperture2 = np.dot(imageApperture,imageApperture)
    
    #print(imageApperture2)
    
    img = convoluteImage(Ix2, imageApperture2)
    
    print(img)
    

lucasKanadaImage = lucasKanada(firstImage, secondImage)

cv2.imwrite(path_to_result_image ,firstImage)




