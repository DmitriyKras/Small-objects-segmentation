from U_Net_utils import get_pairs_from_paths_UAV, get_pairs_from_paths_AERIAL, Unet_DataGenerator, focal_loss
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from U_Net_model import DenseUnet
import config


# Build Unet model
Unet = DenseUnet(INPUT_SHAPE, N_CLASSES)

# Set train and validation generators
if DATASET == "UAV"
    train_pairs, val_pairs = get_pairs_from_paths_UAV(IMAGE_PATH_UAV, MASK_PATH_UAV, 
                                                  IMAGE_PATH_ALL, MASK_PATH_ALL)
if DATASET == "AERIAL"
    train_pairs, val_pairs = get_pairs_from_paths_AERIAL(IMAGE_PATH_AERIAL, MASK_PATH_AERIAL)

# Set up generators
train_gen = Unet_DataGenerator(train_pairs, INPUT_SHAPE[:2], N_CLASSES, STEPS_PER_EPOCH, BATCH_SIZE)
val_gen = Unet_DataGenerator(val_pairs, INPUT_SHAPE[:2], N_CLASSES, 400, BATCH_SIZE)


# Compile model and load weights
Unet.compile(loss=focal_loss,
            optimizer="adam",
            metrics=[Precision(class_id = 1), Recall(class_id = 1)])

if LOAD_WEIGHTS
    Unet.load_weights(WEIGTH_PATH_LOAD)

# Set model checkpoint
CHECPOINT_PATH = WEIGTH_FOLDER_SAVE + "Unet-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(CHECPOINT_PATH, save_weights_only=True, verbose=1, period=5)
callbacks_list = [checkpoint]

# Train model
history = Unet.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                        validation_data = val_gen, callbacks=callbacks_list)
