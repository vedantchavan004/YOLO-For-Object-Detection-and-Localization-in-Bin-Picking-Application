import argparse
import os
import numpy as np
from shafts_preprocessing import read_annotations
from shafts_frontend import ShaftsYOLO
import json
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  #"0" for gpu usage

argparser = argparse.ArgumentParser(
    description='Train and validate vsv-yolo')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


def _main_(args):
    config_path = args.conf
    print( "loading config...", config_path )

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    print( "done\n" )

    ###############################
    #   Parse the annotations
    ###############################

    # parse annotations of the training set and validation set
    train_imgs, train_labels = read_annotations(config['train']['train_image_folder'] )
    valid_imgs, valid_labels = read_annotations(config['valid']['valid_image_folder'] )

    print( "train_imgs= ", train_imgs )

    # random shuffle training and validation data
    np.random.shuffle(train_imgs)
    np.random.shuffle(valid_imgs)

    ###############################
    #   Construct the model
    ###############################

    yolo = ShaftsYOLO( input_width      = config['model']['input_width'],
                        input_height    = config['model']['input_height'],
                        num_channels    = config['model']['num_channels'] )

    ###############################
    #   Start the training process
    ###############################

    print( "enter training" )
    yolo.train(train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               nb_epochs          = config['train']['nb_epochs'],
               learning_rate      = config['train']['learning_rate'],
               batch_size         = config['train']['batch_size'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               direction_scale    = config['train']['direction_scale'],
               saved_weights_name = config['train']['saved_weights_name'],
               debug              = config['train']['debug'],
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'] )

if __name__ == '__main__':
    args = argparser.parse_args()

    #set command line arguments
    args.conf = "shafts_config.json"

    #execute _main_
    _main_(args)
