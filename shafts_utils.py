# Edited Dec-2022 #
import numpy as np
import tensorflow as tf
import cv2
import math
######################### Edit if needed ###############################
obj_threshold=0.8
########################################################################
class KeyPointPair:
    def __init__(self, x0, y0, x1, y1, conf ): #conf= 0.0
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.conf= conf
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.conf # self.classes[self.get_label()]

        return self.score
    
#NMS to reduce multiple detections
def non_max_suppression(boxes, scores, threshold):
    # Initialize a list to store the boxes we want to keep
    kept_boxes = []    
    # Iterate over the boxes and scores
    for i in range(len(boxes)):
        # If the score for this box is lower than the threshold, skip it
        if scores[i] < threshold:
            continue
            
        # Otherwise, add the box to the list
        kept_boxes.append(boxes[i])
       
        # Iterate over the remaining boxes
        for j in range(i+1, len(boxes)):
            # If the boxes overlap significantly, remove the lower scoring box
            if boxes_overlap(boxes[i], boxes[j]):

                # Check if the box is in the list before trying to remove it            
                if scores[i] > scores[j]:
                    if boxes[j] in kept_boxes:                        
                        kept_boxes.remove(boxes[j])
                else:
                    if boxes[i] in kept_boxes:                       
                        kept_boxes.remove(boxes[i])
                    
    # Return the list of boxes we want to keep    
    return kept_boxes

def boxes_overlap(box1, box2):
    # Calculate the coordinates of the intersection of the two boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate the area of the intersection
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate the area of each box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate the IoU
    iou = intersection_area / (box1_area + box2_area - intersection_area)   
    
    # Return True if the IoU is above a threshold, False otherwise
    return iou > 0.5

def draw_kpp(image,kpps):
        
    for kpp in kpps:
        x0 = int(kpp[0])
        y0 = int(kpp[1])        
        x1 = int(kpp[2])
        y1 = int(kpp[3])
        len= math.sqrt((x1-x0)*(x1-x0)+ (y1-y0)*(y1-y0))
        if len>20: #to remove predictions length less than 20             
            cv2.circle(image, (x0,y0), 4, (0,0,255), 1)
            cv2.arrowedLine( image, (x0,y0), (x1,y1), (0,0,255), 1 )    
    return image    

def decode_netout(netout, img_w, img_h): 
    grid_h, grid_w, nb_kpp = netout.shape[:3]
    kpps = []
    scores =[]
    for row in range(grid_h):
        for col in range(grid_w):
            for ikpp in range(nb_kpp):
                # from 4th element onwards are confidence and class classes
                conf = tf.sigmoid(netout[row,col,ikpp,4])

                x0, y0, x1, y1,conf = netout[row,col,ikpp,:5]

                x0 = ((col + (x0)) / grid_w) *img_w
                y0 = ((row + (y0)) / grid_h) *img_h
                x1 = ((col + (x1)) / grid_w) *img_w
                y1 = ((row + (y1)) / grid_h) *img_h                
                
                #Appending netout to the list
                if(x1>0 and x0>0 and y1>0 and y0>0):
                    kpp=[x0,y0,x1,y1]
                    scores.append(conf)
                    kpps.append(kpp)           
    
    kpps=   non_max_suppression(kpps, scores, obj_threshold)   
    
    return kpps

def sigmoid(x):
    return 1. / (1. + np.exp(-x))



