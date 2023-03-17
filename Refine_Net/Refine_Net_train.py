from Refine_Net_utils import get_pairs_from_paths_UAV, get_pairs_from_paths_AERIAL, DataGenerator, focal_loss
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from Refine_Net_model import RefineNet
from config import *


# Build RefineNet model
Refinenet = RefineNet(INPUT_SHAPE, N_CLASSES)

# Set train and validation generators
if DATASET == "UAV":
    train_pairs, val_pairs = get_pairs_from_paths_UAV(IMAGE_PATH_UAV, MASK_PATH_UAV, 
                                                  IMAGE_PATH_ALL, MASK_PATH_ALL)
if DATASET == "AERIAL":
    train_pairs, val_pairs = get_pairs_from_paths_AERIAL(IMAGE_PATH_AERIAL, MASK_PATH_AERIAL)

# Set up generators
train_gen = DataGenerator(train_pairs, INPUT_SHAPE[:2], N_CLASSES, STEPS_PER_EPOCH, BATCH_SIZE)
val_gen = DataGenerator(val_pairs, INPUT_SHAPE[:2], N_CLASSES, 400, BATCH_SIZE)


# Compile model and load weights
if LOSS == "CCE":
    loss = "categorical_crossentropy"
if LOSS == "FCE":
    loss = focal_loss

Refinenet.compile(loss=loss,
            optimizer=OPTIMIZER,
            metrics=[Precision(class_id = 1), Recall(class_id = 1)])

if LOAD_WEIGHTS:
    Refinenet.load_weights(WEIGTH_PATH_LOAD)

# Set model checkpoint
CHECPOINT_PATH = WEIGTH_FOLDER_SAVE + "Refinenet-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(CHECPOINT_PATH, save_weights_only=True, verbose=1, period=5)
callbacks_list = [checkpoint]

# Train model
history = Refinenet.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                        validation_data = val_gen, callbacks=callbacks_list)
