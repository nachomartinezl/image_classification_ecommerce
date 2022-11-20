# Report: Sprint Project 05
> Vehicle classification from images

## 1. The task
The main goal in this project was to build a CNN model capable of predicting car classes appearing in a set of images. As a base model, imagenet weights are used within the structure of a ResNet50 neural network. As a first approach, the parameters obtained by the images selected for the training are positioned in the top layer of the network. Then, having some custom weights as a result, a round of fine tuning is performed in the whole structure unfreezing the layers previously set to non-trainable. At this point, an improvement in the accuracy metrics is expected, but given the images dataset present considerable backgrounds that would feed noise into the training, a removal background step is then applied to the whole dataset, in order to perform another training session from with the cropped images, in order to achieve better validation accuracy scores.

## 2. The data
Data is composed of 196 classes of cars, totaling 16185 images, where 8144 were used for training and 8041 for the test set. In the training step, the samples were split in train and validation, using the 20% for the last one. Cars appear in multiple positions and angles. Data augmentation such as random zoom, height and width is also applied in the model.

## 3. The workflow
Images are downloaded from an S3 bucket using the 'download.py' script. Then, with 'prepare_train_dataset.py' the folders for each class of car are created, inside the separation between train and test folders. All the files are reorganized through this new directories. The 'resnet_50.py' parameters are called from the 'train.py' script, that also obtains parameters for the model from the 'config.yml' file that is loaded in 'utils.py'. Data augmentation step uses the configuration specified in the .yml file and the layers are created in 'data_aug.py' and sent to the model file. For the second part of the training, dataset are processed with the 'remove_background.py' script, that thanks to Detectron2 technology inside 'detection.py' removes everything but the frame of the car in every image.

A docker image is mounted in order to gather all the requirements needed to run the model. Inside the container, GPU provided by AWS performs the computation for every experiment and after every session model weights and logs are stored in a dedicated folder. Predictions are made from the notebook 'Model Evaluation.ipynb', where the performance of the best model is analyzed with a classification report provided by scikit-learn.

## 4. Experiments
In order to achieve the best possible results, 13 experiments has been made:

```
exp_001: imagenet
exp_002: imagenet
├── exp_003: model.12
├── exp_004: model.12
├── exp_005: model.12
│   └── exp_006: model.69
└── exp_008: model.12
exp_007: imagenet
exp_009: imagenet (v2)
    └── exp_010: model.22 (v2)
exp_011: imagenet (v2)
exp_012: imagenet (v2)
    └── exp_013: model.24 (v2)
```
The first eight experiments used the original images for train and test. After achieving an acceptable val_accuracy with no overfitting in exp_002, those weights were used in the next experiments with the base_model unfreezed.
![exp_002 accuracy](/path/to/img.jpg "Accuracy")


## 5. Conclusion




