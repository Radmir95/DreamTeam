import scipy.ndimage as nd
import numpy as np
import cv2


path_to_first_image = "./images/monkey5_001.jpg"
path_to_second_image = "./images/monkey5_002.jpg"

path_to_result_image = "./output/lucasKanada.jpg"

firstImage = cv2.imread(path_to_first_image,cv2.IMREAD_GRAYSCALE)
secondImage = cv2.imread(path_to_second_image,cv2.IMREAD_GRAYSCALE)


def convoluteImage(image, imageFilter):
    
    sizeX, sizeY = image.shape()
    
    
    

def lucasKanada(firstGrayImage, secondGrayImage):
    first = np.array([[1],[4],[6],[4],[1]])
    second = np.array([1,4,6,4,1])
    
    Ix = nd.sobel(firstGrayImage,1)
    Iy = nd.sobel(firstGrayImage,0)
    
    imageFilter = first*second

    convolImage = convoluteImage(firstGrayImage, imageFilter)

lucasKanadaImage = lucasKanada(firstImage, secondImage)

cv2.imwrite(path_to_result_image ,firstImage)




