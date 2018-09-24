import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import keras.backend as K

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,\
                         BatchNormalization, Input, concatenate, \
                         GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from models import ResNet_mnist, Expert_Net17_mnist, DenseNet_mnist

def load_data(split=True):
    """Load the mnist data from the csv files.
    ----------
    split: Boolean
        Set to true to perform the train_test_split
    Returns
    -------
    X_train: array
        Training dataset. Size (N, 28, 28, 1), N being fixed by the split.
    Y_train: array
        Training target. Size (N, 10)
    X_val: array
        Validation dataset. Size (N, 28, 28, 1), N being fixed by the split.
    Y_val: array
        Validation target. Size (N, 10)
    ---------
    """
    # Load the data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    Y_train = train["label"]
    # Drop 'label' column
    X_train = train.drop(labels = ["label"],axis = 1)
    # free some space
    del train
    X_train = X_train / 255.0
    test = test / 255.0
    X_train = X_train.values.reshape(-1,28,28,1)
    test = test.values.reshape(-1,28,28,1)
    Y_train = to_categorical(Y_train, num_classes = 10)
    if split:
        return train_test_split(X_train, Y_train, test_size = 0.1)
    return X_train, Y_train

if __name__ == '__main__':

    X_train, X_val, Y_train, Y_val = load_data()

    #choose a model to train
    """
    model_resnet = ResNet_mnist()
    model_resnet.train(X_train, X_val, Y_train, Y_val)
    model_resnet.test()
    """
    """
    model_densenet = DenseNet_mnist()
    model_densenet.train(X_train, X_val, Y_train, Y_val)
    model_densenet.test()
    """
    X_train, Y_train = load_data(split=False)
    model_expert17 = Expert_Net17_mnist()
    model_expert17.train(X_train, Y_train)
    model_expert17.test()
