# EEL840 FML Project - Team Akshar


## Notes
* Optional arguments are prefixed with '!'
* Please remove '[' and ']' from arguments when attempting to run any program using the following guide.

## Programs description

### 1. train.py
This program can be used to train a model using two datasets- training data and its associated labels.

#### Usage:
```
python ./train.py [train_data_path] [train_labels_path] [!standardize]
```

##### Description:
* train_data_path: (string) <br>
The path to the training dataset
* train_labels_path: (string) <br>
The path to the labels of training dataset
* standardize: (optional) (True|False) <br>
The boolean controls whether to standardize and preprocess data before training. Default value is True.

##### Outputs:
The program on completion saves:
* the trained model in './models' folder.
* the lists for training and validation accuracies and losses in ./metrics folder. These files are required to generate Accuracy, Loss, and Confusion Matrix visualizations.

### 2. test.py
This program can be used to evaluate a trained model on a test dataset. *If test_labels_path is provided, the program will generate a classification report.*

#### Usage:
```
python ./test.py [trained_model_path] [test_data_path] [!test_labels_path] [!standardize]
```

##### Description:
* trained_model_path: (string) <br>
The path to the trained model
* test_data_path: (string) <br>
The path to the test dataset
* test_labels_path: (optional) (string) <br>
The path to the labels of the test dataset
* standardize: (optional) (True|False) <br>
The boolean controls whether to standardize and preprocess data before evaluating. Default value is True.

##### Outputs:
The program on completion saves:
* predicted_labels in './results'. This file contains characters class labels. For example, a, b, et cetera.
* predicted_classes in './results'. This file contains numerical class labels. For example, 1, 2, et cetera.
* a classification report if test_labels_path is provided.

## Configurations
All the configurations are defined in ./modules/globals.py. <br>
For instance,
* No. of epochs
* Learning rate
* Kernel size
* Batch size
* Programs defaults
* Path variables, et cetera