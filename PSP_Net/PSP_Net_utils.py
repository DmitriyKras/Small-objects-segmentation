
# Import all modules for data providing
import glob
import itertools
import os
import random
from sklearn.model_selection import train_test_split

# Import all modules for losses and training
import skimage.measure
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from math import exp, isnan, pow, ceil
import tensorflow as tf
import tensorflow.keras.backend as K


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
  
  
class CPA_DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for CPAnet 
    yields X, main_output, mask_output
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
        
        
    def get_mask(self, path, reshape=True):
        "Load mask from path"
        seg_label = np.zeros((self.input_shape[0], self.input_shape[1], 
                               self.num_classes))  # create label array
        mask = cv2.imread(path, 1)  # load mask
        mask = cv2.resize(mask, self.input_shape, 
                          interpolation=cv2.INTER_NEAREST)  # resize mask
        mask = mask[:, : , 0]  # get one chanel mask

        for c in range(self.num_classes):
            seg_label[:, :, c] = (mask == c).astype(int)  # fill label array
        
        if reshape == False:
            return seg_label  # return without reshaping
        
        return seg_label.reshape((self.input_shape[0] * self.input_shape[1], 
                                  self.num_classes))
      
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
            mask = self.get_mask(pair[1], reshape=True)
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
        return (images, 
                {"main_output_activation": masks, "mask_output_activation": masks})
      
      
def categorical_focal_loss_with_iou(gamma=2., alpha=.25, model=None):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    
    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # print(iou)
        # return the intersection over union value
        return iou

    
    def get_region_props(image: np.ndarray):
        im = image.copy()

        if len(im.shape) == 3 and im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        label_image = skimage.measure.label(im)
        region_props = skimage.measure.regionprops(label_image)
        return region_props

    
    def get_iou_score(y_true_reshaped, y_pred_reshaped):
        props_im = get_region_props(y_pred_reshaped)
        props_gt = get_region_props(y_true_reshaped)

        IOU_bbx_mul = np.zeros((props_gt.__len__(), props_im.__len__()))

        # returning -1 only if there is no gt bbox found
        if len(props_gt) == 0:
            return -1

        for g_b in range(0, props_gt.__len__()):
            for p_b in range(0, props_im.__len__()):
                IOU_bbx_mul[g_b, p_b] = bb_intersection_over_union(props_gt[g_b].bbox, props_im[p_b].bbox)

        row_ind, col_ind = linear_sum_assignment(1 - IOU_bbx_mul)

        calculated_IoU = []
        for ir in range(0, len(row_ind)):
            IOU_bbx_s = IOU_bbx_mul[row_ind[ir], col_ind[ir]]

            calculated_IoU.append(IOU_bbx_s)

            # if IOU_bbx_s >= 0.5:
            #     TP = TP + 1
            # else:
            #     FP = FP + 1
            #     # FN = FN + 1
            #     FP_loc = 1
        # if (props_im.__len__() - props_gt.__len__()) > 0:
        #     FP = FP + (props_im.__len__() - props_gt.__len__())
        #     FP_loc = 1

        if len(calculated_IoU) > 0:
            calculated_IoU_mean = np.mean(calculated_IoU)
        else:
            calculated_IoU_mean = 0.0

        if isnan(calculated_IoU_mean):
            calculated_IoU_mean = 0.0

        if calculated_IoU_mean < 0:
            calculated_IoU_mean = 0.0

        return calculated_IoU_mean

    
    def compute_iou(y_true, y_pred):
        IoUs = []
        # iterating over batch
        for i in range(y_true.shape[0]):
            y_true_single = y_true[i, :, :].reshape(model.output_width, model.output_height, 2).argmax(axis=2)
            y_pred_single = y_pred[i, :, :].reshape(model.output_width, model.output_height, 2).argmax(axis=2)

            IoU = get_iou_score(y_true_single, y_pred_single)
            if isnan(IoU):
                IoU = 0

            # only add IoU score if it is not -1
            if IoU != -1:
                IoUs.append(IoU)

        if len(IoUs) > 0:
            return float(np.mean(IoUs))
        else:
            return -1

        
    def loss_from_iou(y_true, y_pred):
        average_iou = compute_iou(y_true, y_pred)

        if average_iou >= 0.8 or average_iou == -1:
            loss = 0
        else:
            loss = exp(1 - average_iou)

        return float(loss)

    
    def get_center_of_bbox(mask_arr):
        mask_arr_region_props = get_region_props(mask_arr)

        centers = []
        for reg_prop in mask_arr_region_props:
            x1, y1, x2, y2 = reg_prop.bbox
            X = int(np.average([x1, x2]))
            Y = int(np.average([y1, y2]))

            centers.append({'x': X, 'y': Y})
        return centers

    
    def loss_for_difference_in_center(y_true, y_pred):
        total_distance = 0

        # iterating over batch
        for i in range(y_true.shape[0]):
            y_true_single = y_true[i, :, :].reshape(model.output_width, model.output_height, 2).argmax(axis=2)
            y_pred_single = y_pred[i, :, :].reshape(model.output_width, model.output_height, 2).argmax(axis=2)

            y_true_centers = get_center_of_bbox(y_true_single)
            y_pred_centers = get_center_of_bbox(y_pred_single)

            # Continue loop for next iteration if bbox for prediction or ground truth isn't found
            if len(y_true_centers) == 0 or len(y_pred_centers) == 0:
                continue

            center_losses = np.zeros((len(y_true_centers), len(y_pred_centers)), dtype=np.float)

            for i in range(len(y_true_centers)):
                y_true_x = y_true_centers[i]['x']
                y_true_y = y_true_centers[i]['y']

                for j in range(len(y_pred_centers)):
                    y_pred_x = y_pred_centers[j]['x']
                    y_pred_y = y_pred_centers[j]['y']

                    center_losses[i, j] = pow(y_true_x - y_pred_x, 2) + pow(y_true_y - y_pred_y, 2)

            for i in range(center_losses.shape[0]):
                total_distance += np.min(center_losses[i, :])

        total_distance *= 0.5
        return np.float(total_distance)

    
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # added IoU calculation code at start in order to decrease CPU, GPU jump
        iou_loss_val = tf.numpy_function(loss_from_iou, inp=[y_true, y_pred], Tout=tf.double)
        iou_loss_val.set_shape((1,))
        iou_val = tf.cast(iou_loss_val, dtype=tf.float32)

        loss_for_diff = tf.numpy_function(loss_for_difference_in_center, inp=[y_true, y_pred], Tout=tf.double)
        loss_for_diff.set_shape((1,))
        loss_for_difference = tf.cast(loss_for_diff, dtype=tf.float32)

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.math.reduce_sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        # return K.sum(loss, axis=1)
        f_loss = tf.math.reduce_sum(loss, axis=1)

        return f_loss + iou_val + loss_for_difference

    return categorical_focal_loss_fixed


def smooth_l1_loss(y_true, y_pred):
    return tf.keras.losses.Huber()(y_true, y_pred)



def focal_loss(y_true, y_pred):
     # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= tf.math.reduce_sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    alpha = 0.25
    gamma = 2

        # Calculate Cross Entropy
    cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
    loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        # return K.sum(loss, axis=1)
    f_loss = tf.math.reduce_sum(loss, axis=1)
    return f_loss   
