#import libraries for pre-processing
import numpy as np
import cv2
import os

def roı_clahe_pre_process(folder,new_folder):
    
    for filename in os.listdir(folder):
        
        img = cv2.imread(os.path.join(folder,filename),0) #read image from directory
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert rgb to gray scala for apply threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #apply binary and otsu threshold
        x,y,w,h = cv2.boundingRect(thresh) #determine boundary of ROI
        x, y, w, h = x, y, w+20, h+20 #+20 pixels tolerans for boundaries
        img = img[y:y+h, x:x+w] #crop original image with the help of boundary
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #determine clahe values
        img = clahe.apply(img) #apply clahe transform on image
        
        new_path=new_folder+filename #determine new path for save to image 
        cv2.imwrite(new_path, img) #save output to new paths
        
#determine source and target folder paths
folder="C:/Users/User/Desktop/mura_dataset/mura/val/1"
new_folder="C:/Users/User/Desktop/mura_dataset_clahe/mura/val/1/"

#run funnction with folder paths
roı_clahe_pre_process(folder,new_folder)
