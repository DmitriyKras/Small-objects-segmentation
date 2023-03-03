from PSP_Net_utils import get_pairs_from_paths_UAV, get_pairs_from_paths_AERIAL, CPA_DataGenerator, smooth_l1_loss, focal_loss, categorical_focal_loss_with_iou
from PSP_Net_model import CPA_net
from tensorflow.keras.metrics import Precision, Recall, IoU
from tensorflow.keras.callbacks import ModelCheckpoint


# Model params
INPUT_SHAPE = (473, 473)
BATCH_SIZE = 4
EPOCHS = 10
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

# Weight path
WEIGTH_PATH = "/kaggle/input/cpa-weights/UAV-CPA-CCE-20.h5"

# Build CPA model
CPA = CPA_net(N_CLASSES, 50, INPUT_SHAPE, activation='softmax')

# Set train and validation generators
train_pairs, val_pairs = get_pairs_from_paths_UAV(IMAGE_PATH_UAV, MASK_PATH_UAV, 
                                                  IMAGE_PATH_ALL, MASK_PATH_ALL)
#train_pairs, val_pairs = get_pairs_from_paths_AERIAL(IMAGE_PATH_AERIAL, MASK_PATH_AERIAL)
train_gen = CPA_DataGenerator(train_pairs, INPUT_SHAPE[:2], N_CLASSES, STEPS_PER_EPOCH, BATCH_SIZE)
val_gen = CPA_DataGenerator(val_pairs, INPUT_SHAPE[:2], N_CLASSES, 400, BATCH_SIZE)

# Compile model and load weights
CPA.compile(loss={
            "main_output_activation": "categorical_crossentropy",
            "mask_output_activation": smooth_l1_loss,
        },
            optimizer="adam",
            metrics={"main_output_activation" : [Precision(class_id = 1), Recall(class_id = 1)],
                     "mask_output_activation" : ["accuracy"]})

CPA.load_weights(WEIGTH_PATH)

CHECKPOINT_PATH = "/kaggle/working/CPA-CCE-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True, verbose=1, period=5)
callbacks_list = [checkpoint]

# Train model
history = CPA.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                        validation_data = val_gen, callbacks=callbacks_list)
