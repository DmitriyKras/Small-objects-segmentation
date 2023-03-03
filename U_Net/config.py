# Model params
INPUT_SHAPE = (512, 512, 3)
BATCH_SIZE = 4
EPOCHS = 20
OPTIMIZER = "adam"
STEPS_PER_EPOCH = 1000
N_CLASSES = 2


# Datasets pathes examples (choose dataset)
DATASET = "AERIAL"  # = "UAV"

# UAV Dataset pathes
IMAGE_PATH_UAV = "/kaggle/input/drone-detection-dataset/crops_with_target/crops_with_target"
MASK_PATH_UAV = "/kaggle/input/drone-detection-dataset/masks_with_target/masks_with_target"

IMAGE_PATH_ALL = "/kaggle/input/drone-detection-dataset/images_cropped/images_cropped"
MASK_PATH_ALL = "/kaggle/input/drone-detection-dataset/masks_cropped/masks_cropped"

# AERIAL Dataset pathes
IMAGE_PATH_AERIAL = "/kaggle/input/drone-detection-dataset/crops_with_target/crops_with_target"
MASK_PATH_AERIAL = "/kaggle/input/drone-detection-dataset/masks_with_target/masks_with_target"


# Weights path (load)
LOAD_WEIGHTS = False  # = True if you want to resume training or just load weights
WEIGTH_PATH_LOAD = "/kaggle/input/unet-w/Unet-FCE-30.h5"

# Weights folder (save)
WEIGTH_FOLDER_SAVE = ""


