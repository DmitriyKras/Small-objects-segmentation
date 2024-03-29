# Models for small object segmentation


This project contains implementations of different semantic segmentation models trained on 2 datasets with small flying objects (drones, airplanes, helicopters) 

## Inspirations

**Papers:**

[DenseU-Net-Based Semantic Segmentation of Small Objects in Urban Remote Sensing Images](https://ieeexplore.ieee.org/document/8718619)

[Dogfight: Detecting Drones from Drones Videos](https://arxiv.org/abs/2103.17242)

[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

[PSP-Net repo](https://github.com/mwaseema/image-segmentation-keras-implementation)

[RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612)

[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587v3)


## Datasets overview

UAV dataset - [Perdue UAV dataset](https://engineering.purdue.edu/~bouman/UAV_Dataset/)

AERIAL dataset - [Svanström F, Englund C and Alonso-Fernandez F. (2020). Real-Time Drone Detection and Tracking With Visible, Thermal and Acoustic Sensors](https://arxiv.org/pdf/2007.07396.pdf)

**Processed datasets available in my Kaggle profile**

[UAV Dataset](https://www.kaggle.com/datasets/llpukojluct/drone-detection-dataset)

This dataset contains 4 folders: *crops_with_target*, *masks_with_target* and *images_cropped*, *masks_cropped*. Folders *with target* contain images and masks with targets, folders *cropped* contain all images and masks with or without actual target. Mask has a same filename as corresponding image and concsists of 0 at background pixels and 1 at foreground pixels.

[AERIAL Dataset](https://www.kaggle.com/datasets/llpukojluct/aerial)

This dataset has 2 folders: *images* and *masks*. Mask format is same as in UAV Dataset.

## Project overview

Project contains folders named as corresponding model implementation. *utils.py* has all essential functions and classes for data providing and training (collect image-mask pairs from path, data generators and loss functions). *model.py* has model building functions. *train.py* has a script for training and weights saving. *config.py* has config parameters for model building and training such as pathes for dataset, weights etc.

## Data providing

For *get_pairs_from_path_UAV* you need to provide pathes for images and masks with targets and pathes for all images and targets.

For *get_pairs_from_path_AERIAL* you need to provide pathes for images and masks

## Dependancies and usage

**Dependencies:**

TensorFlow = 2.11.0\
OpenCV = 4.6.0\
scikit-image = 0.19.3

**Installation:**
```
git clone https://github.com/DmitriyKras/Small-objects-segmentation.git
cd Small-objects-segmentation
```

**Training:**

Set config variables in `modelname/config.py` and execute `modelname/modelname_train.py`

**Visualize predictions**

Set path to image for predictions in `modelname/modelname_predict.py`, set mode "mask" if you want to see segmented mask or "bbox" if you want to see bounding boxes and then execute file

**Some examples of predictions below**

![Example 1](/predictions_examples/example1.png)
![Example 2](/predictions_examples/example2.png)
![Example 2](/predictions_examples/example3.png)

## Evaluation
U-Net, PSP-Net, Refine-Net and DeepLabV3 were trained on mixed dataset and evaluated on 39 manualy collected and labeled images. 
Because of small amount of test data different augmentation methods were apllied with *imgaug*: affine augmentation (flip and rotation), weather-like augmentation (fog, rain, snow) and
camera-like augmentation (blur, motion blur and additive noise). The impact on perfomance of each augmentation type provided on ROC curve figures.


**Without augmentation**
![Without augmentation](/test_results/ROC_curves_test.png)
**Affine augmentation**
![Affine augmentation](/test_results/ROC_curves_test_affine.png)
**Weather-like augmentation**
![Weather augmentation](/test_results/ROC_curves_test_weather.png)
**Camera-like augmentation**
![Camera augmentation](/test_results/ROC_curves_test_camera.png)
