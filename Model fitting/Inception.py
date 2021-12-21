from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from keras.applications.inception_v3 import InceptionV3
from keras.regularizers import l2
from keras.models import Model
import tensorflow as tf
from DataGenerators import *

################################################################
# Initial model training
################################################################

base_inception = InceptionV3(weights='imagenet', include_top=False,
                             input_shape=(299, 299, 3))

base_inception.trainable = False

# Add a pooling layer
out = base_inception.output
out = GlobalAveragePooling2D()(out)

# add new classification layers
out = Dense(512, activation='relu')(out)
out = Dropout(.3)(out)
out = Dense(256, activation='relu')(out)
out = Dropout(.3)(out)
predictions = Dense(80, kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(.0005), activation='softmax')(out)

model = Model(inputs=base_inception.input, outputs=predictions)

opt = SGD(lr=.00005, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit_generator(train_generator,
                              epochs=30, verbose=1,
                              validation_data=val_generator)

model.save('/kaggle/working/inceptionclassifier.h5')

################################################################
# Model Fine Tuning
################################################################

model = tf.keras.models.load_model(r'/kaggle/input/inception50/inceptionclassifier.h5')

# train all layers
for layer in model.layers:
    layer.trainable = True

opt = SGD(lr=.00005, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              epochs=30, verbose=1,
                              validation_data=val_generator)

model.save('/models/Inception_Tuned.h5')
