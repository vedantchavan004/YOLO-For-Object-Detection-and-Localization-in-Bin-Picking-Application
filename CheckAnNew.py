from typing import List
import ast
import cv2
import math

from tkinter import*
import tkinter as tk
from tkinter import filedialog
from PIL import Image,ImageTk
import os

############################################################################################
                       # Checking Annotations Function #
############################################################################################
def checkimg():

    #Select img
    file_path=filedialog.askopenfilename(initialdir=os.getcwd(),title='Select image file',
                                         filetype=(('BMP file','*.bmp'),('JPG File','*.jpg')))
    img=Image.open(file_path)    
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) #IMREAD_UNCHANGED: without changing resolution

    path,file_=os.path.split(file_path)
    print(file_)    #image_name
    imn= file_.split('.')[0]

    #select .txt file
      
    fn=".txt"
    file_name=imn+fn  
        
    #read file
    with open(file_name) as f:
        Annotation_files = f.read()

        # Copy Of Data From String To List
        dataList = Annotation_files.split('\n')
        print('---------------------------------------')
        print('Total No of Shafts :   ', len(dataList) - 1)  # No. of Shafts
        print('---------------------------------------')

        #array, variables
        a = []     
        c=0     #dist from center to direction
        d=0     #dist from center to direction
        
        #for loop to go through each shaft

        for i in range(0, len(dataList)-1, 1):
            a = str.split(dataList[i])
            x0, y0, x1, y1 = int(float(a[1])), int(float(a[2])), (float(a[3])), (float(a[4]))
            #printing data
            print('X: ',a[1])
            print('Y: ',a[2])
            print('X1: ',a[3])
            print('Y1: ',a[4])
            print('C1: ',a[5])
            
            print('---------------------------------------')                  
        
            #calculation for drawing line considering phi and alpha
            #c= int(20*(math.cos(alpha)*math.cos(phi))) #20 is pixel |can be changed as per liking 
            #d= int(20*(math.sin(alpha)*math.cos(phi)))        
    
            #calculating new coordinates w.r.t. origin of image
            c=int(x1)
            d=int(y1)
        
            #plotting geometry
            cv2.circle(img, (x0, y0), 4, (0,0, 255), 1)     #img,starting coordinates,radius,colour,circle thickness(measured in pixel)
            cv2.line(img, (x0, y0),(c, d) , (0, 0, 255), 1) #img,first corrdinates,second coordinates,colour,line thickness
            cv2.imshow(imn, img)                            #heading for image frame, plotted img

            #to get next data for shaft press 0 or arrow key
            cv2.waitKey(0)

   
        
            
    

################################################################################
                            #GUI#
################################################################################


root=Tk()


fram=Frame(root)
fram.pack(side=BOTTOM,padx=15,pady=15)
lbl=Label(root)
lbl.pack()
btn=Label(root)
lbl.pack()

btn=Button(fram,text='Select Image',command=checkimg)
btn.pack(side=tk.LEFT)

btn2=Button(fram,text='Exit',command=lambda:exit())
btn2.pack(side=tk.LEFT,padx=12)

root.title('CheckImage')
root.geometry('250x50')

path_dir=os.getcwd()
os.chdir(path_dir+'\data')

 


###################################################################################




        
        
