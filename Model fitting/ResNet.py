import tensorflow as tf
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import ResNet50V2
from keras.regularizers import l2
from DataGenerators import *

################################################################
# Initial model training
################################################################

# import model
base = ResNet50V2(weights='imagenet', include_top=False,
                  input_shape=(224, 224, 3))

base.trainable = False

# Add a pooling layer
out = base.output
out = GlobalAveragePooling2D()(out)

# add new classification layers
out = Dense(512, activation='relu')(out)
out = Dropout(.3)(out)
out = Dense(256, activation='relu')(out)
out = Dropout(.3)(out)
predictions = Dense(80, kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(.0005), activation='softmax')(out)

model = Model(inputs=base.input, outputs=predictions)

# train the classification layers
opt = Adam(lr=.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              epochs=30, verbose=1,
                              validation_data=val_generator)

# save model
model.save('/kaggle/working/resnet_v1.h5')

################################################################
# Model Fine Tuning
################################################################

# fine tune parameters in all layers.
model = tf.keras.models.load_model(r'/kaggle/input/resnet/resnet_v1.h5')

for layer in model.layers:
    layer.trainable = True

opt = Adam(lr=.00001)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              epochs=30, verbose=1,
                              validation_data=val_generator)

model.save('/models/ResNet_Tuned.h5')
