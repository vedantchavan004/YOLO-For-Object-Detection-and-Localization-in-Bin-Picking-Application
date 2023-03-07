#########################
#Jan-2023
#########################

from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D,UpSampling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2
from shafts_utils import decode_netout
from keras.applications.mobilenet import MobileNet
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from shafts_preprocessing import YoloBatchGenerator
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, TensorBoard
import keras
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
import keras.backend as K
from tensorflow.python.platform import tf_logging as logging
#from callbacks import CustomModelCheckpoint, CustomTensorBoard


# callback class, use it later maybe for visualization of training progress
class YoloCallback( Callback ):
    def __init__( self, v_epoch_counters ):
        Callback.__init__(self)
        self.v_epoch_counters = v_epoch_counters
    

    def on_epoch_end( self, epoch, logs=None ):
        self.v_epoch_counters[0] = self.v_epoch_counters[0] + 1
        print( "epoch_counter =", self.v_epoch_counters[0] )
         

# construct network structure which is derived from YOLO V2
class ShaftsYOLO(object):
    def __init__(self, input_width,
                       input_height,
                       num_channels ):

        self.input_width = input_width
        self.input_height = input_height
        self.num_channels   = num_channels
        self.v_epoch_counters = [0]

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image    = Input(shape=(self.input_height, self.input_width, self.num_channels))

        num_layer = 0 # intial layer assigned as 0

        # stack 1
        

        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(input_image)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        num_layer += 1

        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        num_layer += 1

        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        num_layer += 1

        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1
        #x = MaxPooling2D(pool_size=(2, 2))(x)
        
        #upsampling
        x = UpSampling2D()(x)
        

        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1

        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1

        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        num_layer += 1

        #stack4

        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1
    

        #stack5

        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1

        #stack6
 
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1

        #stack7

        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1

        #stack8

        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_' + str(num_layer), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1

        #final

        x = Conv2D(5, (3,3), strides=(1,1), padding='same', name='conv_'+str( num_layer ), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str( num_layer ))(x)
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1

        print( "x.shape=", x.shape.as_list() )
        self.grid_h = x.shape.as_list()[1]
        self.grid_w = x.shape.as_list()[2]

        print( "self.grid_h, self.grid_w=", self.grid_h, self.grid_w )

        # reshape output layer
        # it contains grid_h * grid_w predictor cells
        output = Reshape((self.grid_h, self.grid_w, 1, 5))(x) 

        print( "model_1 input shape=", input_image.shape )
        print( "model_2 output shape=", output.shape )

        self.model = Model(inputs=input_image, outputs=output)

        #--------------------------------------------------------------------------------        
        #self.model.load_weights( "shafts.h5" )  # load pretrained model, deactivate by comment if model structure has been changed
        #--------------------------------------------------------------------------------


        # print a summary of the whole model
        self.model.summary(positions=[.25, .60, .80, 1.])
        tf.compat.v1.logging.set_verbosity(tf.logging.INFO)  ## testein

    #custom loss function defines the tensorflow graph to compute the final_loss value
    # which is needed for training process
    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4] # mask_shape is then (batch_size, gridsize_x, gridsize_y, 1)

        # cell_x and cell_y are x and y coordinates, one for each cell, this is a basic lookup table for fast coordinate access in training loop
        cell_x = tf.cast( tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)), dtype=tf.float32 )  #same dimension number as y_pred, contains grid cell x coordinates
        cell_y = tf.cast( tf.reshape( tf.transpose( tf.reshape( tf.tile( tf.range(self.grid_h), [self.grid_w] ),(self.grid_w, self.grid_h))),(1, self.grid_h, self.grid_w, 1, 1)), dtype=tf.float32 )#same dimension number as y_pred, contains grid cell y coordinates


        # cell_grid contains x and y coordinates for each batch and for each grid cell, this is a lookup table for fast coordinate access in training loop
        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, 1, 1]) # has now x and y coordinates ascending in one dimension [[x,x,x,x,y,y,y,y],[x,x,x,x,y,y,y,y],...]...

        # masks are used for fast loss computation
        # Intialize all masks with 0
        coord_mask = tf.zeros(mask_shape, dtype='float32')#used to compute the coordinates of the keypoints
        conf_mask  = tf.zeros(mask_shape, dtype='float32')#Used to tell the confidence of the located keypoints
        class_mask = tf.zeros(mask_shape, dtype='float32')#Used to identify the class of the object

        total_recall = tf.Variable(0., dtype='float32')  #used in status report only

        # Extract  p r e d i c t i o n  x1y1, x2y2, confidence and class from y_pred
        #pred_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid  # given in grid coordinate system
        pred_xy = (y_pred[..., :2]) + cell_grid  # given in grid coordinate system

        #pred_x2y2 = tf.sigmoid(y_pred[..., 2:4]) + cell_grid # Changed predicted area moments, shape is (batch_size, gridsize_x, gridsize_y, nb_anchors, <2,3 == Ixx, Ixy))
        pred_x2y2 = (y_pred[..., 2:4]) + cell_grid # Changed predicted area moments, shape is (batch_size, gridsize_x, gridsize_y, nb_anchors, <2,3 == Ixx, Ixy))

        ### extract (=limit to [0...1]) predicted confidence
        # pred_conf = y_pred[...,4]
        pred_conf = tf.sigmoid(y_pred[..., 4]) #predicted confidences shape is (batch_size, gridsize_x, gridsize_y, 1, [4==confidence]))

        
        #Extract ground truth x1,y1, x2, y2, conf from y_true
        
        ### keypoint0 x y alpha
        true_xy = y_true[..., 0:2] # x-y-coordinates of y_true, in grid coordinate system
        #true_xy = tf.sigmoid(y_true[..., :2]) + cell_grid 
        true_x2y2 = y_true[..., 2:4] 
        #true_am= tf.sigmoid(y_true[..., 2:4]) + cell_grid
        ### confidence
        # true_conf = y_true[..., 4]
        true_conf = tf.sigmoid(y_true[..., 4]) # confidence
        
        # Compose the masks for loss calculation and punishing
        
        # this is the confidence for each keypoint, multiplied by the coord_scale.
        # extract 4th axis only, this is <confidence>,
        # and append a dimension with size 1 at the end -> (nb_batches, gridsize_x, gridsize_y, 1 )
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        # at the end conf_mask has all elements set either to no_object_scale or to object_scale
        # penalize the confidence difference of all keypoints which are farer away from true keypoints
        # set conf_mask to 1 where no objects are in ground_truth and to 0 where objects are in ground_truth 
        conf_mask = conf_mask + 1.0 - y_true[...,4] 
        conf_mask = conf_mask * self.no_object_scale  # set all mask ones to no_object_scale
        #conf_mask.shape==(nb_batches, nb_grid_x, nb_grid_y, nb_anchors)


        # penalize the confidence difference of all keypoints which are reponsible for corresponding ground truth objects, consider y_true[...,4] is either 1 or 0
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale  # set the cells containing keypoints to object_scale

        # compute the loss
        # numbers are needed for a kind of normalization
        nb_coord_kp = tf.reduce_sum(tf.cast(coord_mask > 0.0, dtype=tf.float32)) # number of grid cells containing objects
        nb_conf_kp  = tf.reduce_sum(tf.cast(conf_mask  > 0.0, dtype=tf.float32)) # number of all grid cells
        
        loss_xy = tf.reduce_sum(tf.square(true_xy-pred_xy) * coord_mask) / (nb_coord_kp + 1e-6)
        loss_x2y2 = tf.reduce_sum(tf.square(true_x2y2-pred_x2y2) * coord_mask ) / (nb_coord_kp + 1e-6)
        loss_conf  = tf.reduce_sum(tf.square(true_conf-pred_conf) * conf_mask)  / (nb_conf_kp + 1e-6)

        loss = loss_xy + loss_conf + loss_x2y2

        if self.debug:
            nb_true_kp = tf.reduce_sum(y_true[..., 4])
            nb_pred_kp = tf.reduce_sum(tf.cast(true_conf > 0.5, dtype=tf.float32) * tf.cast(pred_conf > 0.5, dtype=tf.float32))

            current_recall = nb_pred_kp/(nb_true_kp + 1e-6)
            total_recall = total_recall + current_recall          

        
        return loss

    def get_f1(self, y_true, y_pred): #shapes (batch, 4)
               
        true_conf = tf.sigmoid(y_true[..., 4])
        pred_conf = tf.sigmoid(y_pred[..., 4])
        ground_positives = tf.reduce_sum(y_true[..., 4]) + 1e-6
        #if true conf and pred_conf>0.5 
        true_positives = tf.reduce_sum(tf.cast(true_conf > 0.5, dtype=tf.float32) * tf.cast(pred_conf > 0.5, dtype=tf.float32)) + 1e-6
        pred_positives = tf.reduce_sum(y_pred[..., 4]) + 1e-6
       
    
        precision = true_positives / (pred_positives + 1e-6)
        recall = true_positives / (ground_positives + 1e-6)
        #both = 1 if ground_positives == 0 or pred_positives == 0
        

        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        #still with shape (4,)

        weighted_f1 = f1 * ground_positives / tf.reduce_sum(ground_positives) 
        weighted_f1 = tf.reduce_sum(weighted_f1)

    
        return weighted_f1 #for metrics, return only 'weighted_f1'

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)
        self.model.save(weight_path+"full")

        print( "input layer name=" )
        print( [node.op.name for node in self.model.inputs] )
        print( "output layer name=" )
        print( [node.op.name for node in self.model.outputs] )


        return self.model.output.shape[1:3]

    def normalize(self, image):
        return image / 255.


    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to validate the model
                    nb_epochs,      # number of epoches
                    learning_rate,  # the learning rate
                    batch_size,     # the size of the batch
                    object_scale,
                    no_object_scale,
                    coord_scale,
                    direction_scale,
                    saved_weights_name='transparent.h5',
                    debug=True,
                    train_times=3,
                    valid_times=1):

        self.batch_size = batch_size
        self.object_scale    = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale     = coord_scale
        self.direction_scale = direction_scale
        self.debug = debug

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H'         : self.input_height,
            'IMAGE_W'         : self.input_width,
            'GRID_H'          : self.grid_h,
            'GRID_W'          : self.grid_w,
            'CHANNELS'        : self.num_channels,
            'BATCH_SIZE'      : self.batch_size,
        }

        train_generator = YoloBatchGenerator(train_imgs,
                                     generator_config,
                                     norm=self.normalize)
        valid_generator = YoloBatchGenerator(valid_imgs,
                                     generator_config,
                                     norm=self.normalize,
                                     jitter=False)

        ############################################
        # Compile the model
        ############################################

        
        
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer, metrics=[self.get_f1, 'acc'])

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=30,#2or3
                           mode='min',
                           verbose=1)

        checkpoint = ModelCheckpoint(saved_weights_name,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only = False,
                                     mode='min',
                                     period=1)

        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                                  histogram_freq=1,
                                  write_graph=True,
                                  write_images=True,
                                  update_freq='epoch',
                                  profile_batch=2)

                         

        lr_reducer = ReduceLROnPlateau(monitor="val_loss",
                                       factor=0.8,
                                       patience=5,
                                       verbose=1,
                                       mode='min',
                                       min_delta=0.0001,
                                       cooldown=3,
                                       min_lr=1e-6)
        
        

        yolo_callback = YoloCallback( self.v_epoch_counters )

        ############################################
        # Start the training process
        ############################################
        print( "call fit_generator" )
        self.model.fit_generator(generator        = train_generator,
                                 steps_per_epoch  = len(train_generator)*train_times,
                                 epochs           = nb_epochs,
                                 verbose          = 2 if debug else 1,
                                 validation_data  = valid_generator,
                                 validation_steps = len(valid_generator) * valid_times,
                                 callbacks        = [early_stop, checkpoint, tensorboard, yolo_callback, lr_reducer],
                                 workers          = 3,
                                 max_queue_size   = 8)
                                 #use_multiprocessing = False)


    def predict(self, image):
        #print("image.shape=",image.shape)
        image_h, image_w, _  = image.shape
        image = self.normalize(image)

        input_image = image[:,:,::-1] #flip rgb to bgr or vice versa
        input_image = np.expand_dims(input_image, 0)

        netout = self.model.predict([input_image])[0]# add dummy_array

        np.set_printoptions(threshold=sys.maxsize)
        #print( "netout=", [netout] )  # print the netout

        netout_decoded = decode_netout(netout, image_w, image_h)

        #for kpp in netout_decoded:
           # print( kpp.x0, kpp.y0, kpp.ixx, kpp.ixy, kpp.conf  )

        return netout_decoded


        
     
