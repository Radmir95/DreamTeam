import cv2
import numpy as np
import matplotlib.pyplot as plt 

xAxis = 0
yAxis = 1
path_to_image = './images/lena_color_512.tif'


grayImage = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
npGrayImage = np.uint8(grayImage)


mean = np.mean(npGrayImage)
variance = np.var(npGrayImage)
median = np.median(npGrayImage)

print('Mean: ', mean)
print('Variance: ', variance)
print('Median: ', median)


binarizationImage = np.zeros(npGrayImage.shape)

for xPx in range(npGrayImage.shape[xAxis]):
    for yPx in range(npGrayImage.shape[yAxis]):
        if npGrayImage[xPx, yPx] > median:
            binarizationImage[xPx][yPx] = 1
        else:
            binarizationImage[xPx][yPx] = 0
                                 

cv2.imshow('Binarization Image', binarizationImage)
cv2.waitKey(0)
cv2.destroyAllWindows()


plt.xlabel('Histogram')
plt.hist(npGrayImage,bins=5)
plt.show()


integralImage = np.zeros(npGrayImage.shape)

for i in range(npGrayImage.shape[xAxis]):
    sum = 0
    for j in range(npGrayImage.shape[yAxis]):
        sum+= npGrayImage[i][j]
        if (i == 0):
            integralImage[i][j] = sum
        else:
            integralImage[i][j] = integralImage[i-1][j] + sum
                    
cv2.imshow('Integral Image', integralImage)
cv2.waitKey(0)
cv2.destroyAllWindows()