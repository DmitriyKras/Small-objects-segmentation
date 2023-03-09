from PSP_Net_utils import get_pairs_from_paths_UAV, get_pairs_from_paths_AERIAL, CPA_DataGenerator, smooth_l1_loss, focal_loss, categorical_focal_loss_with_iou
from PSP_Net_model import CPA_net
from tensorflow.keras.metrics import Precision, Recall, IoU
from tensorflow.keras.callbacks import ModelCheckpoint
from config import *


# Build CPA model
CPA = CPA_net(INPUT_SHAPE, N_CLASSES)

# Set train and validation generators
if DATASET == "UAV":
    train_pairs, val_pairs = get_pairs_from_paths_UAV(IMAGE_PATH_UAV, MASK_PATH_UAV, 
                                                  IMAGE_PATH_ALL, MASK_PATH_ALL)
if DATASET == "AERIAL":
    train_pairs, val_pairs = get_pairs_from_paths_AERIAL(IMAGE_PATH_AERIAL, MASK_PATH_AERIAL)

# Set up generators
train_gen = CPA_DataGenerator(train_pairs, INPUT_SHAPE[:2], N_CLASSES, STEPS_PER_EPOCH, BATCH_SIZE)
val_gen = CPA_DataGenerator(val_pairs, INPUT_SHAPE[:2], N_CLASSES, 400, BATCH_SIZE)

# Compile model and load weights
if LOSS == "CCE":
    loss = "categorical_crossentropy"
if LOSS == "FCE":
    loss = focal_loss
if LOSS == "FCE_IoU":
    loss = categorical_focal_loss_with_iou

CPA.compile(loss={
            "main_output_activation": loss,
            "mask_output_activation": smooth_l1_loss,
        },
            optimizer="adam",
            metrics={"main_output_activation" : [Precision(class_id = 1), Recall(class_id = 1)],
                     "mask_output_activation" : ["accuracy"]})

CPA.load_weights(WEIGTH_PATH)

# Set model checkpoint
CHECPOINT_PATH = WEIGTH_FOLDER_SAVE + "CPA_net-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(CHECPOINT_PATH, save_weights_only=True, verbose=1, period=5)
callbacks_list = [checkpoint]

# Train model
history = CPA.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                        validation_data = val_gen, callbacks=callbacks_list)
