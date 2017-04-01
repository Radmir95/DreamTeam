from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
import cv2

inputfile = './images/chessBoard.jpg'
outputfile = './output/chessBoard.jpg'
image_with_circle_path = './images/coins.jpg'
image_with_circle_outputpath = 'Circle_Detected_Image.jpg'

def houghLines(img,angle,threshold):
	
	y_idxs, x_idxs = np.nonzero(img)
	thetas = angle*(np.arange(-(np.pi/(angle*2)), (np.pi/(angle*2))))
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


def houghLineFound(path_to_file,outputfile):
    
    color_image = cv2.imread(path_to_file, cv2.IMREAD_COLOR)
    gray_image = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
    edges = canny_edge_detection(gray_image)
    lines = houghLines(edges,(np.pi)/180,100)
    resultImage =  drawAndReturnLines(color_image, lines)
    return resultImage

def drawAndReturnLines(color_image,lines):
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
		
        cv2.line(color_image,(x1,y1),(x2,y2),(0,0,255), 1)
	
    return color_image

def canny_edge_detection(image):
    
    image = np.int32(image)

    otsu_threshold_val, ret_matrix = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    lower_threshold = otsu_threshold_val * 0.4
    upper_threshold = otsu_threshold_val * 1.3

    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    
    return edges
    
def houghCircles(image): 
    
    sizeX = image.shape[0] 
    sizeY = image.shape[1] 
    
    sinang = dict() 
    cosang = dict() 
    
    circles = []
    
    startRadius = 5

    for angle in range(0,360): 
        sinang[angle] = np.sin(angle * np.pi/180) 
        cosang[angle] = np.cos(angle * np.pi/180) 
            
    length=int(sizeX/2)
    
    radius = [i for i in range(startRadius,length)]
       
    for r in radius:
        acc_cells = np.full((sizeX,sizeY),fill_value=0,dtype=np.uint64)
         
        for x in range(sizeX): 
            for y in range(sizeY): 
                if image[x][y] == 255:
                    for angle in range(0,360): 
                        b = y - round(r * sinang[angle]) 
                        a = x - round(r * cosang[angle]) 
                        if a >= 0 and a < sizeX and b >= 0 and b < sizeY: 
                            acc_cells[a][b] = acc_cells[a][b] + 1
                             
        print('For radius: ',r)
        acc_cell_max = np.amax(acc_cells)
        print('max acc value: ',acc_cell_max)
        
        if(acc_cell_max > 150):     
            
            acc_cells[acc_cells < 150] = 0  

            for i in range(sizeX): 
                for j in range(sizeY): 
                    if(i > 0 and j > 0 and i < sizeX-1 and j < sizeY-1 and acc_cells[i][j] >= 150):
                        avg_sum = np.float32((acc_cells[i][j]+acc_cells[i-1][j]+acc_cells[i+1][j]+acc_cells[i][j-1]+acc_cells[i][j+1]+acc_cells[i-1][j-1]+acc_cells[i-1][j+1]+acc_cells[i+1][j-1]+acc_cells[i+1][j+1])/9) 
                        if(avg_sum >= 33):
                            circles.append((i,j,r))
                            acc_cells[i:i+5,j:j+5] = 0
        return circles
                    
def houghCircleFound(output_file_path):
    orig_img = cv2.imread(image_with_circle_path)
        
    image = cv2.imread(image_with_circle_path,cv2.IMREAD_GRAYSCALE) 
    
    input_img = deepcopy(image)
        
    edged_image = canny_edge_detection(input_img)
    
    #add here optimization
    
    
    circles = houghCircles(edged_image)  
    
    for vertex in circles:
        cv2.circle(orig_img,(vertex[1],vertex[0]),vertex[2],(0,255,0),1)
        cv2.rectangle(orig_img,(vertex[1]-2,vertex[0]-2),(vertex[1]-2,vertex[0]-2),(0,0,255),3)
            
    cv2.imwrite(output_file_path, orig_img) 
    
    
linesImage = houghLineFound(inputfile,outputfile)

cv2.imwrite(outputfile,linesImage)

houghCircleFound(image_with_circle_outputpath)
