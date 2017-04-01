import cv2
import time
import numpy as np
import scipy
from scipy.ndimage.filters import convolve

path_to_lena = "./images/lena_color_512.tif"

sx = np.int32([[-1,0,1],[-2,0,2],[-1,0,1]])
sy = np.int32([[1,2,1],[0,0,0],[-1,-2,-1]])


def scipyEdgeDetector(image):
    
    image = np.int32(image)
    dx = convolve(image,sx) 
    dy = convolve(image,sy)
    magnitude = np.power(np.power(dx, 2.0) + np.power(dy, 2.0), 0.5)
    magnitude *= 255.0 / np.max(magnitude)  # normalize
    scipy.misc.imsave('scipyEdgeDetector.jpg', magnitude)
    

def loopEdgeDetector(image):
    
    image = np.int32(image)
    sizeX, sizeY = image.shape

    #make border
    imageZeros = np.zeros((sizeX+2, sizeY+2))
    
    for sizeX in range(1, sizeX+1):
        for sizeY in range(1, sizeY+1):
            imageZeros[sizeX][sizeY] = image[sizeX-1][sizeY-1]
    
    imageZeros = np.int32(imageZeros)
    
    resultDx = []
    resultDy = []
    
    for rectY in range(0, sizeY):
        
        arrDx = []
        arrDy = []
        for rectX in range(0, sizeX):
            img = imageZeros[rectY:rectY+3,rectX:rectX+3]       
            multDx = np.sum(np.dot(img, sx))
            multDy = np.sum(np.dot(img, sy))
            arrDx.append(multDx)
            arrDy.append(multDy)
        resultDx.append(arrDx)
        resultDy.append(arrDy)
    
    resultDx = np.int32(arrDx)
    arrDy = np.int32(arrDy)
    
    resultDx = np.int32(resultDx)
    resultDy = np.int32(resultDy)
    
    magnitude = np.power(np.power(resultDx, 2.0) + np.power(resultDy, 2.0), 0.5)

    magnitude *= 255.0 / np.max(magnitude)  # normalize
    scipy.misc.imsave('loopEdgeDetector.jpg', magnitude)

def numpyEdgeDetector(img):
    n,m = img.shape
    
    ax = img[0:n-2,0:m-2]*sx[0,0]
    bx = img[0:n-2,1:m-1]*sx[0,1]
    cx = img[0:n-2,2:m]*sx[0,2]
    dx = img[2:n,0:m-2]*sx[2,0]
    ex = img[2:n,1:m-1]*sx[2,1]
    fx = img[2:n,2:m]*sx[2,2]
    ay = img[0:n-2,0:m-2]*sy[0,0]
    by = img[1:n-1, 0:m-2] * sy[1, 0]
    cy = img[2:n, 0:m-2] * sy[2, 0]
    dy = img[0:n-2, 2:m] * sy[0, 2]
    ey = img[1:n - 1, 2:m] * sy[1, 2]
    fy = img[2:n, 2:m] * sy[2, 2]
    
    dx = ax + bx + cx + dx + ex + fx
    dy = ay + by + cy + dy + ey + fy
    
    magnitude = np.power(np.power(dx, 2.0) + np.power(dy, 2.0), 0.5)
    magnitude *= 255.0 / np.max(magnitude)
    
    scipy.misc.imsave('numpyEdgeDetector.jpg', magnitude)


image = cv2.imread(path_to_lena, cv2.IMREAD_GRAYSCALE)

print("**** Scipy edge detector ****")
start = time.time()
scipyEdgeDetector(image)
end = time.time()
print(end - start)
print("****    ****")

print("**** Loop edge detector ****")
start = time.time()
loopEdgeDetector(image)
end = time.time()
print(end - start)
print("****    ****")

print("**** Numpy edge detector ****")
start = time.time()
numpyEdgeDetector(image)
end = time.time()
print(end - start)
print("****    ****")