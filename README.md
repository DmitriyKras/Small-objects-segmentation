# Models for small object segmentation


This project contains implementations of different semantic segmentation models trained on 2 datasets with small flying objects (drones, airplanes, helicopters) 

## Inspirations

**Papers:**

[DenseU-Net-Based Semantic Segmentation of Small Objects in Urban Remote Sensing Images](https://ieeexplore.ieee.org/document/8718619)

[Dogfight: Detecting Drones from Drones Videos](https://arxiv.org/abs/2103.17242)

[PSP-Net repo](https://github.com/mwaseema/image-segmentation-keras-implementation)


## Datasets overview

UAV dataset - [Perdue UAV dataset](https://engineering.purdue.edu/~bouman/UAV_Dataset/)

AERIAL dataset - [Svanström F, Englund C and Alonso-Fernandez F. (2020). Real-Time Drone Detection and Tracking With Visible, Thermal and Acoustic Sensors](https://arxiv.org/pdf/2007.07396.pdf)

**Processed datasets available in my Kaggle profile**

[UAV Dataset](https://www.kaggle.com/datasets/llpukojluct/drone-detection-dataset)

This dataset contains 4 folders: *crops_with_target*, *masks_with_target* and *images_cropped*, *masks_cropped*. Folders *with target* contain images and masks with targets, folders *cropped* contain all images and masks with or without actual target. Mask has a same filename as corresponding image and concsists of 0 at background pixels and 1 at foreground pixels.

[AERIAL Dataset](https://www.kaggle.com/datasets/llpukojluct/aerial)

This dataset has 2 folders: *images* and *masks*. Mask format is same as in UAV Dataset.


Set pathes in train files and change compile params

Datasets links:
AERIAL  https://www.kaggle.com/datasets/llpukojluct/aerial
UAV  https://www.kaggle.com/datasets/llpukojluct/drone-detection-dataset
