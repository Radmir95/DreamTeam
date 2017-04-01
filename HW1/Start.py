import cv2
import numpy as np
import matplotlib.pyplot as plt 

xAxis = 0
yAxis = 1
cv2BlueChannel = 0
cv2GreenChannel = 1
cv2RedChannel = 2
path_to_image = './images/lena_color_512.tif'

def showImage(label='Image', image=None):
    if image:
        cv2.imshow(label, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
            
def binarizationImage(image):
    binarizationImage = np.zeros(image.shape)

    for xPx in range(image.shape[xAxis]):
        for yPx in range(image.shape[yAxis]):
            if image[xPx, yPx] > median:
                binarizationImage[xPx][yPx] = 1
            else:
                binarizationImage[xPx][yPx] = 0
    
    return binarizationImage

grayImage = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
colorImage = cv2.imread(path_to_image, cv2.IMREAD_COLOR)

npGrayImage = np.uint8(grayImage)
npColorImage = np.uint8(colorImage)

mean = np.mean(npGrayImage)
variance = np.var(npGrayImage)
median = np.median(npGrayImage)

print('Mean: ', mean)
print('Variance: ', variance)
print('Median: ', median)

binarizationImage = binarizationImage(npGrayImage)
showImage('Binarization Image', binarizationImage)

integralImage = integralImage(npGrayImage)
showImage('Integral Image', integralImage)

plt.xlabel('Histogram')
plt.hist(npGrayImage,bins=5)
plt.show()       

blue = npColorImage[:,:,cv2BlueChannel]
green = npColorImage[:,:,cv2GreenChannel]
red = npColorImage[:,:, cv2RedChannel]

plt.figure()
plt.title('Histogram GBR')
plt.hist(blue,bins = 5)
plt.hist(green,bins = 5)
plt.hist(red,bins = 5)

#bin = np.uint8(img > tr) return boolean variable
#h, bin = np.hist()
# we can made plt.plot(bin[:-1],h) - graphic of histogram more comfortable way