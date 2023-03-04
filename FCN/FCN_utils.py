import cv2
import tensorflow as tf
import os
import random
from sklearn.model_selection import train_test_split
import numpy as np


def get_pairs_from_paths_AERIAL(images_path, segs_path, val_size=0.2):
    "Get list of pairs image-label from folder for AERIAL VEHICLE dataset"
    images = os.listdir(images_path)
    segmentations = os.listdir(segs_path)
    images = list(set(images) & set(segmentations))
    random.shuffle(images)
    pairs = [(os.path.join(images_path, img), os.path.join(segs_path, img)) for img in images]
    return train_test_split(pairs, test_size=val_size)
  
  
def get_pairs_from_paths_UAV(images_path, segs_path, images_path_all, segs_path_all,
                         val_size=0.2):
    images = os.listdir(images_path)
    segmentations = os.listdir(segs_path)
    images_target = list(set(images) & set(segmentations))
    
    images = os.listdir(images_path_all)
    segmentations = os.listdir(segs_path_all)
    images_all = list(set(images) & set(segmentations) - set(images_target))
    
    random.shuffle(images_all)
    images = images_target + images_all[:3*len(images_target)]
    
    pairs = [(os.path.join(images_path_all, img), os.path.join(segs_path_all, img)) for img in images]
    
    return train_test_split(pairs, test_size=val_size)
  
  
class FCN_DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for CPAnet 
    yields X, main_output
    """
    def __init__(self, pairs, input_shape, num_classes, 
                 steps_per_epoch=1000, batch_size = 32):
        self.pairs = pairs  # list with pairs of image-mask
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.steps_per_epoch = steps_per_epoch  # less for validation
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch
    
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        random.shuffle(self.pairs)
        
        
    def get_mask(self, path):
        "Load mask from path"
        seg_label = np.zeros((self.input_shape[0], self.input_shape[1], 
                               self.num_classes))  # create label array
        mask = cv2.imread(path, 1)  # load mask
        mask = cv2.resize(mask, self.input_shape, 
                          interpolation=cv2.INTER_NEAREST)  # resize mask
        mask = mask[:, : , 0]  # get one chanel mask

        for c in range(self.num_classes):
            seg_label[:, :, c] = (mask == c).astype(int)  # fill label array
            
        return seg_label  # return without reshaping
      
      
    def get_image(self, path):
        "Load image from path"
        img = cv2.imread(path, 1)  # load mask
        img = cv2.resize(img, self.input_shape)  # resize mask
        img = img.astype(np.float32)  # convert to float
        img /= 255.0  # normalize image
        return img
    
    
    def get_batch(self, batch_pairs):
        "Get batch of input image and output masks"
        images = []  # create empty lists for images and masks
        masks = []
        for pair in batch_pairs:  # iterate over batch pairs
            img = self.get_image(pair[0])  # load image and mask
            mask = self.get_mask(pair[1])
            images.append(img)  # append to lists
            masks.append(mask)
        images = np.array(images)  # convert to arrays
        masks = np.array(masks)
        return images, masks
    
    
    def __getitem__(self, index):
        "Get item - batch with index"
        batch_pairs = self.pairs[index * self.batch_size : 
                                 (index + 1) * self.batch_size]  # get batch pairs
        images, masks = self.get_batch(batch_pairs)  # get batch
        return images, masks
      
      
def focal_loss(y_true, y_pred):
     # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= tf.math.reduce_sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    alpha = 0.25
    gamma = 3

        # Calculate Cross Entropy
    cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
    loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        # return K.sum(loss, axis=1)
    f_loss = tf.math.reduce_sum(loss, axis=1)
    return f_loss