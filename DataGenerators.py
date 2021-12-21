import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
IMG_HEIGHT = 224
IMG_WIDTH = 224

# read labels
train_labels = pd.read_csv('/data/train_labels.csv', dtype={'img_name': str, 'label': str})

# make data generator objecct that feeds data to network
datagen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=.1
)

# Generator for the training set
train_generator = datagen.flow_from_dataframe(
    dataframe=train_labels,
    directory='/data/train_set/train_set/',
    x_col='img_name',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    subset='training',
    batch_size=40
)

# generator for the testing set
val_generator = datagen.flow_from_dataframe(
    dataframe=train_labels,
    directory='/data/train_set/train_set/',
    x_col='img_name',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    subset='validation'
)
