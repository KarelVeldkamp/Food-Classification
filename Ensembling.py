import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

IMG_HEIGHT = IMG_WIDTH = 224


# Function hta returns dataframe of predictions based on models,
# model weights, a data generator and a dataframe of test images
def predict_test_set(models, generator, df):
    # calculate weighted sum of output layers
    preds = np.zeros(80)
    for i, model in enumerate(models):
        preds += model.predict(generator) * weights[i]

    # map max weighted sum to label
    preds = np.argmax(preds, axis=1)
    submission = df

    # loop though predictions, convert them to proper labels and add to dataframe
    for i in range(len(preds)):
        print(submission['label'][i] == out2class[preds[i]], flush=True)
        submission.at[i, 'label'] = out2class[preds[i]]

    # return dataframe of predictions
    return submission


# read test set dataframe
sample = pd.read_csv("/data/sample.csv", dtype={'img_name': str, 'label': str} )

# image generators
datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = datagen.flow_from_dataframe(
    dataframe=sample,
    directory= "C:/Users/kavel/Documents/data science master/AML/food-recognition-challenge-2021/test_set/test_set/" ,
    x_col='img_name',
    y_col='label',
    class_mode = 'categorical',
    shuffle=False,
    target_size=(IMG_HEIGHT,IMG_WIDTH)
)

train_generator = datagen.flow_from_dataframe(
    dataframe=labels,
    directory= "/kaggle/input/foodrec/train_set/train_set/",
    x_col='img_name',
    y_col='label',
    class_mode = 'categorical',
    shuffle=False,
    target_size=(IMG_HEIGHT,IMG_WIDTH)
)

# load model
dense = tf.keras.models.load_model('/models/DenseNet_Tuned.h5')
inception = tf.keras.models.load_model('/models/Inception_Tuned.h5')
mobilenet = tf.keras.models.load_model('/models/MobileNet_Tuned.h5')
resnet = tf.keras.models.load_model('/models/ResNet_Tuned.h5')
xception = tf.keras.models.load_model('/models/Xception_Tuned.h5')

# dictionary that maps network output to class labels
class_names = train_generator.class_indices
out2class = {v: k for k, v in class_names.items()}

# the model and their weights, which are their validation accuracies
models = [dense, inception, mobilenet, resnet, xception]
weights = [67, 63, 61, 58, 65]

submission = predict_test_set(models, weights, test_generator, sample)

submission.to_csv('ensemble_predictions.csv')



