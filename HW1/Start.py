import cv2

image = cv2.imread('C:/Users/Radmir.XTREME95/Desktop/ComputerVision/HW1/images/lena_color_512.tif', -1)

cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()