# Edited Dec-2022 #

import argparse
import os
import cv2
import numpy as np
#from tqdm import tqdm
from shafts_preprocessing import read_annotations
from shafts_utils import draw_kpp
from shafts_frontend import ShaftsYOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = ShaftsYOLO( input_width  = config['model']['input_width'],
                input_height  = config['model']['input_height'],
                num_channels   = config['model']['num_channels'])

    ###############################
    #   Load trained weights
    ###############################

    grid_height, grid_width = yolo.load_weights(weights_path)
    #print( "grid_height, grid_width=", grid_height, grid_width )

    ###############################
    #   Load image and predict poses
    ###############################

    image = cv2.imread(image_path)
    #imagem = cv2.bitwise_not(image)  #uncomment when using negative image model
    image = image[:,:,1] #extract green channel only because it´s a RGB but grayscale image
    image = np.expand_dims( image, -1 ) #add an empty dimension to get a one-channel shape (height, width, 1)
    #imagem = imagem[:,:,1] #uncomment when using negative image model
    #imagem = np.expand_dims( imagem, -1 ) #uncomment when using negative image model

    kpp = yolo.predict(image) # compute netout

    image = np.concatenate( (image, image, image), axis = 2 ) #create a 3-channel grayscale image

    image = draw_kpp(image, kpp) # draw an image with annotation symbols included, with confidence threshold

    print(len(kpp), 'keypoints are found')
    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)


if __name__ == '__main__':
    args = argparser.parse_args()

    #set command line arguments
    
    args.conf = "shafts_config.json"
    args.input = "images/test/Img_00065.bmp" #Testing img
    args.weights = "shafts.h5"

    #execute _main_
    _main_(args)
