import tensorflow as tf
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.applications import DenseNet169
from keras.models import Model
from DataGenerators import *

################################################################
# Initial model training
################################################################

# Dense Net model pretrained on imagenet
model = DenseNet169(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# we will only train the classification layers
model.trainable = False

# add classification layers with dropout
model_output = model.output
layers = GlobalAveragePooling2D()(model_output)
layers = Dense(512, activation='relu')(layers)
layers = Dropout(.4)(layers)
layers = Dense(256, activation='relu')(layers)
layers = Dropout(.4)(layers)
layers = Dense(80, activation='softmax')(layers)

# compile and fit model
model = Model(inputs=model.input, outputs=layers)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=.00015),
              metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=8,
                    validation_data=val_generator)

# save output
model.save('DenseNet1_v2.h5')


################################################################
# Model Fine Tuning
################################################################

model = tf.keras.models.load_model('/kaggle/input/densenet2/DenseNet1_v2.h5')

# we will now train all layers
for layer in model.layers:
    layer.trainable = True

# low learning rate in order to not lose pre-learned representations
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=.00001),
              metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=15,
                    validation_data=val_generator)

model.save('/models/DenseNet_Tuned.h5')
