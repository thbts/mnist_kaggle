#TODO
import training
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

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
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1)
        return X_train, X_val, Y_train, Y_val, test
    return X_train, Y_train, test

if __name__ == '__main__':
    size_ensemble = 10
    models = []
    results = None
    X_train, X_val, Y_train, Y_val, test = load_data()
    data = (X_train, X_val, Y_train, Y_val, test)
    for i in range(size_ensemble):
        name_model = str('densenet_'+str(i))
        result_test = training.train(name_model, data)
        if results is None:
            results = result_test
        else:
            results += result_test
    
    results = np.argmax(results,axis = 1)
    results = pd.Series(results,name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    submission.to_csv('densenets_ensemble.csv',index=False)
    
