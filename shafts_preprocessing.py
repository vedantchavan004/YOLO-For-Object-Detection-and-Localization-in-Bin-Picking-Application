#Edited Dec-2022
import os
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import imgaug as ia
from keras.utils import Sequence
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from shafts_utils import draw_kpp
import math

def read_annotations(img_dir):
    # annotation file structure (class is always 0):
    #    0   1  2    3     4       5    
    # class, x, y, x1, y1, conf_cylindrical_area
    all_imgs = []
    seen_labels = {}

    for ann_file_name in sorted(os.listdir(img_dir)):
        ext = os.path.splitext( ann_file_name )[1]

        if ext == '.txt':
            img = {'object':[]}

            img_file_name = img_dir + ann_file_name
            print("img_file_name=", img_file_name)
            print( "ann_file_name=", ann_file_name )

            # read in predefined order
            file = open( img_file_name, "r" )

            for line in file:  # each line is a keypoint pair to this picture
                vWords = line.split()
                obj = {}

                img['filename'] = os.path.splitext( img_file_name )[0] + ".bmp"  # image file name
                obj['name'] = vWords[0]  # class name

                if obj['name'] in seen_labels:
                    seen_labels[obj['name']] += 1
                else:
                    seen_labels[obj['name']] = 1

                if float(vWords[5])> 0.5: #conf>0.5
                    obj['x0'] = float( vWords[1] ) 
                    obj['y0'] = float( vWords[2] )     
                    obj['x1'] = float( vWords[3] ) 
                    obj['y1'] = float( vWords[4] ) 

                    img['object'] += [obj]  #this is all annotation data for this image

            file.close()

            if len(img['object']) > 0:
                all_imgs += [img]
    return all_imgs, seen_labels

class YoloBatchGenerator(Sequence):
    def __init__(self, images,
                       config,
                       shuffle=True,
                       jitter=True,
                       norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm
        self.image_counter = 0

        ia.seed( 1 )


        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.8, aug)

        # Define augmentation here
        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                sometimes( iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -10 to +10 percent (per axis)
                    rotate=(-10, 10), # rotate by -10 to +10 degrees
                    #shear=(-5, 5), # shear by -5 to +5 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top in imgaug documentation for examples)
                    mode = "edge"

                )),
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 3 and 11
                        ]),
                        #    iaa.CoarseSalt(0.01, size_percent=(0.002, 0.01)),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0), # randomly remove from 1% up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-15, 15), per_channel=0), # change brightness of images (by -15 to 15 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0), # change brightness of images (50 to 150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ])

            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)


    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))


    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['x0'], obj['y0'], obj['x1'], obj['y1'], 0]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):  # get a complete batch, invoked from training thread
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        # <batchsize> <gridsize_x>, <gridsize_y>, <1>, <obj_x, obj_y, obj_x1, obj_y1, confidence> == desired network output tensor
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], 1, 5))

        i_img = 0
        keypoints_on_images = [] # all keypoints in the image, ready for augmentation
        images_batch = []

        #create list of keypoints, 
        num_images = len( self.images )
        instance_src_index = l_bound % num_images

        #do augmentation
        for instance_count in range( r_bound - l_bound ):
            train_instance = self.images[instance_src_index]
            
            # augment input image and fix object's position and size
            image_name = train_instance['filename']
            img = cv2.imread(image_name)
            img = img[:,:,1]  # green channel only
            img = np.expand_dims( img, -1 )  # reattach a dimension

            images_batch.append( img )

            # construct output from object's x, y, x1, y1
            keypoints_on_image = []
            all_objs = train_instance['object']
            for obj in all_objs:
                keypoints_on_image.append( ia.Keypoint( x=float(obj['x0']), y=float(obj['y0']) ))
                keypoints_on_image.append( ia.Keypoint( x=float(obj['x1']), y=float(obj['y1']) ))
                                
            keypoints_on_images.append(ia.KeypointsOnImage(keypoints_on_image, shape=img.shape ))
            instance_src_index = (instance_src_index + 1) % num_images

        if self.jitter:
            ia.seed( 134 )
            aug_pipe_det = self.aug_pipe.to_deterministic() # so that the augmentation of the images and the keypoints effect the same transformations
            x_batch = aug_pipe_det.augment_images(images_batch) # augmented images
            keypoints_batch_aug = aug_pipe_det.augment_keypoints( keypoints_on_images )  # augmented keypoints
        else:
            x_batch = images_batch
            keypoints_batch_aug = keypoints_on_images

        x_batch = np.reshape( x_batch, (r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 1 ) )
        x_batch = self.norm( x_batch )

        # enter augmented keypoints in y_batch
        num_images = len( self.images )
        instance_src_index = l_bound % num_images
        for instance_count in range( r_bound - l_bound ):
            train_instance = self.images[instance_src_index]
            all_objs = train_instance['object']
            obj_count = 0
            for obj in all_objs:

                # decode augmentated data
                kp0_x = keypoints_batch_aug[instance_count].keypoints[obj_count*2].x
                kp0_x = kp0_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                kp0_y = keypoints_batch_aug[instance_count].keypoints[obj_count*2].y
                kp0_y = kp0_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])
                kp1_x = keypoints_batch_aug[instance_count].keypoints[obj_count*2+1].x
                kp1_x = kp1_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                kp1_y = keypoints_batch_aug[instance_count].keypoints[obj_count*2+1].y
                kp1_y = kp1_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                #grid_x1 = int(np.floor(kp1_x)) #changed
                #grid_y1 = int(np.floor(kp1_y))  #changed
                                                           
                # Determine the grid cell to which the keypoint belongs.
                grid_x = int(np.floor(kp0_x))  #these are the grid coordinates, e.g. in the 4x4 grid into which the image is divided
                grid_y = int(np.floor(kp0_y))

                if grid_x >= 0 and grid_y >= 0 and grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    best_sector= 0  # in this version we have only one sector per grid cell, this can be expanded later
                    y_batch[instance_count, grid_y, grid_x, best_sector, 0] = kp0_x  #keypoint0 in grid-Koordinaten LUC
                    y_batch[instance_count, grid_y, grid_x, best_sector, 1] = kp0_y
                    y_batch[instance_count, grid_y, grid_x, best_sector, 2] = kp1_x #changed
                    y_batch[instance_count, grid_y, grid_x, best_sector, 3] = kp1_y #changed
                    y_batch[instance_count, grid_y, grid_x, best_sector, 4] = 1.  #confidence, is always 1.0 in gound truth

                    self.image_counter += 1

                obj_count += 1
            instance_src_index = (instance_src_index + 1) % num_images


        return x_batch, y_batch   #image normalized and y_batch in grid coordinates

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)  #shuffle along the first axis only
