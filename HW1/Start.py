import cv2
import matplotlib.pyplot as plt 
import numpy as np

xAxis = 0
yAxis = 1

image = cv2.imread('./images/lena_color_512.tif', cv2.IMREAD_GRAYSCALE)

imageArray = np.uint8(image)

print(imageArray)

print(image.shape)

mean = np.mean(imageArray)

variance = np.var(imageArray)

median = np.median(imageArray)

print('Mean: ', mean)
print('Variance: ', variance)
print('Median: ', median)

binarizationArray = np.zeros(imageArray.shape)

for xPx in range(imageArray.shape[xAxis]):
    for yPx in range(imageArray.shape[yAxis]):
        if imageArray[xPx, yPx] > median:
            binarizationArray[xPx][yPx] = 1
        else:
            binarizationArray[xPx][yPx] = 0
                                 
print(binarizationArray)

cv2.imshow('Image', binarizationArray)

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(imageArray,bins=5)
plt.show()
