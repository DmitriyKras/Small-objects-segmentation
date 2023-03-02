from utils import get_pairs_from_paths_UAV, get_pairs_from_paths_AERIAL, Unet_DataGenerator, focal_loss
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from model import DenseUnet

# Model params
INPUT_SHAPE = (512, 512, 3)
BATCH_SIZE = 4
EPOCHS = 20
OPTIMIZER = SGD
STEPS_PER_EPOCH = 1000
N_CLASSES = 2

# UAV Dataset pathes
IMAGE_PATH_UAV = "/kaggle/input/drone-detection-dataset/crops_with_target/crops_with_target"
MASK_PATH_UAV = "/kaggle/input/drone-detection-dataset/masks_with_target/masks_with_target"

IMAGE_PATH_ALL = "/kaggle/input/drone-detection-dataset/images_cropped/images_cropped"
MASK_PATH_ALL = "/kaggle/input/drone-detection-dataset/masks_cropped/masks_cropped"

# AERIAL Dataset pathes
IMAGE_PATH_AERIAL = "/kaggle/input/drone-detection-dataset/crops_with_target/crops_with_target"
MASK_PATH_AERIAL = "/kaggle/input/drone-detection-dataset/masks_with_target/masks_with_target"

# Weights path
WEIGTH_PATH = "/kaggle/input/unet-w/Unet-FCE-30.h5"

# Build Unet model
Unet = DenseUnet(INPUT_SHAPE, N_CLASSES)

# Set train and validation generators
train_pairs, val_pairs = get_pairs_from_paths_UAV(IMAGE_PATH_UAV, MASK_PATH_UAV, 
                                                  IMAGE_PATH_ALL, MASK_PATH_ALL)
#train_pairs, val_pairs = get_pairs_from_paths_AERIAL(IMAGE_PATH_AERIAL, MASK_PATH_AERIAL)
train_gen = Unet_DataGenerator(train_pairs, INPUT_SHAPE[:2], N_CLASSES, STEPS_PER_EPOCH, BATCH_SIZE)
val_gen = Unet_DataGenerator(val_pairs, INPUT_SHAPE[:2], N_CLASSES, 400, BATCH_SIZE)


# Compile model and load weights
Unet.compile(loss=focal_loss,
            optimizer="adam",
            metrics=[Precision(class_id = 1), Recall(class_id = 1)])
Unet.load_weights(WEIGTH_PATH)

# Set model checkpoint
CHECKPOINT_PATH = "/kaggle/working/Unet-CCE-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True, verbose=1, period=5)
callbacks_list = [checkpoint]

# Train model
history = Unet.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                        #validation_data = val_gen, callbacks=callbacks_list)
