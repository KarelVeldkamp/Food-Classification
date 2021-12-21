from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from keras.regularizers import l2
from tensorflow.keras.applications import Xception
from keras.models import Model
import tensorflow as tf
from DataGenerators import *

################################################################
# Initial model training
################################################################

pretrained = Xception(include_top=False, weights='imagenet',
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

pretrained.trainable = False

# Add a pooling layer
out = pretrained.output
out = GlobalAveragePooling2D()(out)

# add new classification layers
out = Dense(512, activation='relu')(out)
out = Dropout(.3)(out)
out = Dense(256, activation='relu')(out)
out = Dropout(.3)(out)
predictions = Dense(80, kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(.0005), activation='softmax')(out)

model = Model(inputs=pretrained.input, outputs=predictions)

history = model.fit_generator(train_generator,
                              epochs=25, verbose=1,
                              validation_data=val_generator)

model.save('/kaggle/working/xnceptionmodel2.h5')

################################################################
# Model Fine Tuning
################################################################

model = tf.keras.models.load_model('/kaggle/input/epoch30/xnceptionmodel.h5')
for layer in model.layers:
    layer.trainable = True
opt = Adam(lr=.0003)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              epochs=25, verbose=1,
                              validation_data=val_generator)

model.save('/models/Xception_tuned.h5')
