import cv2
import scipy
from math import pi, floor, cos, sin
import numpy as np

path_to_image = "./images/i.jpg"

def HoughLines(image,angle,threshold):
    lines = []
    y_idxs, x_idxs = np.nonzero(image)
    thetas = angle*(np.arange(-(np.pi/(angle*2)), (np.pi/(angle*2))))
    print(thetas)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
	
    all_lines = {}
    for t in range(len(thetas)):
        for i in range(y_idxs.size):
            r = round(x_idxs[i]*cos_t[t] + y_idxs[i]*sin_t[t])
            if (r,thetas[t]) in all_lines.keys():
                all_lines[(r,thetas[t])] = all_lines[(r,thetas[t])] + 1
            else:
                all_lines[(r,thetas[t])] = 1	
    all_lines = {k:v for (k,v) in all_lines.items() if v >= threshold}
    return np.asarray(list(all_lines.copy().keys()))

src = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
dst = cv2.Canny(src, 50, 200, 3)
lines = HoughLines(dst,(np.pi)/180,100)
for line in lines:
    rho = line[0]
    theta = line[1]
    a = np.cos(theta)
    b = np.sin(theta)
		
    x0 = a*rho
    y0 = b*rho
		
    x1 = int(np.round(x0 + 1000*(-b)))
    y1 = int(np.round(y0 + 1000*(a)))
    x2 = int(np.round(x0 - 1000*(-b)))
    y2 = int(np.round(y0 - 1000*(a)))
		
    cv2.line(src,(x1,y1),(x2,y2),(0,255,255), 1)
	
cv2.imwrite("./images/iii.jpg",src)