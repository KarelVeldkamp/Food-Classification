# Food recognition with an esemble of transfer leanring models

This is code for an in-class kaggle competition concerned with classifying images of food. The data consists of 30612 training images with labels belonging to one of 80 food categories, and 7653 test images without labels. Labels for the testset are not available, but the accuracy of predictions could be retrieved on kaggle. 


![Random images](/random_images.png "Random Images")

# Model fitting
The final predictions attained a test set accuracy of 72%. This accuracy was reached by using a weighted ensemble of five pre trained convolutional networks. This section summarises our approach. 

#### Data preprocessing and augmentation
All images were resised to 244x244 and normalised to a scale of zero to one. To augment the data, a random width and heihgt shift was sampled from U[0, 0.2]. Images were rotated with the degrees of rotation being sampled from 
U[0,25]. Finally, images were horizontally flipped with a probability of .5. 
10% of the training set was safed for validation purposes. 


#### Tranfer learning
Five pretrained models were imported from [the keras website](https://keras.io/api/applications/), along with the weights trained on the imagenet dataset:
- DenseNet169
- InceptionV3
- MobileNet
- ResNet50v2
- Xception

I added several classifier layers to each of the five models, with dropout and l2 regularisation. In a first round of training for each model, the original convolutional layers were not trained, and only these classifier layeys were being trained. Then, after reaching a reasonable accuracy, I trained the entire emodels with a lower learning rate. This resulted in accuracies between 58 and 67 percent. 

#### Ensemble

To decrease the varaince of predictions on the test set, the weighted sum of the 5 output layers was calculated, and for each image the maximum node of this weighted sum was taken as the prediction of the model. This resulted in an improvement of the accuracy to 72%. 

#### Explainability

To better understand the model, we used LIME, which is framework for model explainability that allows you to display which part of an image contribute most to its classification. LIME works by creating different versions of an image where different patches are blacked out, and looking at the effect of these masks on the classification confidence of the model. 


![LIME plots for inception model](/lime_images.png "LIME plots for inception model")

It looks like the model is generally looking in the right place. The foods itself are mostly highlighted on the different images. However, other objects also seem to influence the calssification. For example, different parts of plates and dishes seem to have an effect on the predictions of the model. This could possibly be because certain food items often cooccur with certain types of food. Also, we see some images where seamingly random, irrelevant parts of the image have a large effect on the predictions. This in an indication of high variance. Making predictions based on multiple models can have lower this variance slightly, which is why the ensemble had a higher accuracy (72%) than the individal models. 
