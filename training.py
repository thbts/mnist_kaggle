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


def train(name_file, data):

    X_train, X_val, Y_train, Y_val, test = data

    #choose a model to train
    """
    model_resnet = ResNet_mnist()
    model_resnet.train(X_train, X_val, Y_train, Y_val)
    model_resnet.test()
    """
    model_densenet = DenseNet_mnist()
    model_densenet.train(X_train, X_val, Y_train, Y_val)
    results = model_densenet.test(test,name_file)
    """
    X_train, Y_train = load_data(split=False)
    model_expert17 = Expert_Net17_mnist()
    model_expert17.train(X_train, Y_train)
    model_expert17.test()
    """
