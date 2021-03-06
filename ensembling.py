#TODO
import training
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

def ensembling_existing_models(model_names, X_test):
    results = None
    for name in model_names:
        model = load_model(name)
        print('model loaded')
        result_test = model.predict(X_test)
        print('prediction done')
        if results is None:
            results = result_test
        else:
            results += result_test
    results = np.argmax(results,axis = 1)
    results = pd.Series(results,name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    submission.to_csv('ensembling_submission.csv',index=False)

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
    size_ensemble = 5
    models = []
    results = None
    classes = (0,1,2,3,4,5,6,7,8,9)
    X_train, X_val, Y_train, Y_val, test = load_data()
    # X_train, Y_train, test = load_data(split=False)
    print('data loaded')

    data = (X_train, X_val, Y_train, Y_val, test)
    # data = (X_train, Y_train, test)
    for i in range(size_ensemble):
        name_model = str('basic_net_rmsprop_val_split'+str(i))
        print(name_model)
        result_test = training.train(name_model, data)
        if results is None:
            results = result_test
        else:
            results += result_test
    """
    #Test the models

    model = load_model('densenet_0.h5')
    model.summary()
    results = model.predict(X_val)
    results = np.argmax(results,axis = 1)
    Y_val = np.argmax(Y_val,axis = 1)
    print('model loaded')
    # Confusion
    matrix = confusion_matrix(Y_val, results, classes)
    matrix = matrix/matrix.sum(axis=0)[None,:]
    print(np.around(matrix, decimals = 4))

    results = np.argmax(results,axis = 1)
    results = pd.Series(results,name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    submission.to_csv('basic_nets_ensemble.csv',index=False)
    """
    model_names = ['basic_net_rmsprop_0.h5',
                    'basic_net_rmsprop_1.h5',
                    'basic_net_rmsprop_2.h5',
                    'basic_net_rmsprop_3.h5',
                    'basic_net_rmsprop_4.h5',
                    'shallower_densenet_rmsprop_0.h5',
                    'shallower_densenet_rmsprop_1.h5',
                    'shallower_densenet_rmsprop_2.h5',
                    'shallower_densenet_rmsprop_3.h5',
                    'shallower_densenet_rmsprop_4.h5']
    ensembling_existing_models(model_names,test)
