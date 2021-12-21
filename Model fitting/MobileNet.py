################################################################
# Imports and Settings
################################################################

from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from keras.regularizers import l2
from tensorflow.keras.applications import MobileNet
from keras.models import Model
import tensorflow as tf
from DataGenerators import *

################################################################
# Initial model training
################################################################

pretrained = MobileNet(include_top=False, weights='imagenet',
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))


pretrained.trainable = False

# Add a pooling layer
out = pretrained.output
out = GlobalAveragePooling2D()(out)

# Add classification layers
out = Dense(512, activation='relu')(out)
out = Dropout(.3)(out)
out = Dense(256, activation='relu')(out)
out = Dropout(.3)(out)
predictions = Dense(80, kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(.0005), activation='softmax')(out)

model = Model(inputs=pretrained.input, outputs=predictions)

history = model.fit_generator(train_generator,
                              epochs=30, verbose=1,
                              validation_data=val_generator)

model.save('/kaggle/working/mobilenet1.h5')

################################################################
# Model Fine Tuning
################################################################

model = tf.keras.models.load_model('/kaggle/input/mobilenet/mobilenet1.h5')

for layer in model.layers:
    layer.trainable = True


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=.00001),
              metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=20,
                    validation_data=val_generator)

model.save('/models/MobileNet_Tuned.h5')
