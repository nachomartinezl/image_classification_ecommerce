# Report: Sprint Project 05
> Vehicle classification from images

## 1. The task
The main goal in this project was to build a CNN model capable of predicting car classes appearing in a set of images. As a base model, imagenet weights are used within the structure of the ResNet50 neural network. As a first approach, the parameters obtained by the images selected for the training are positioned in the top layer. Then, having some custom weights as a result, a round of fine tuning is performed in the whole structure unfreezing the layers previously set to non-trainable. At this point, an improvement in the metrics is expected, but given the images dataset present considerable backgrounds that would feed noise into the training, a removal background step is then applied to the whole dataset, in order to perform another training session from with the cropped images, in order to achieve better validation accuracy scores.

## 2. The data

## 3. Technologies

## 4. The workflow

## 5. Experiments

## 6. Conclusion




