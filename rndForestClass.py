from collections import Counter
from sklearn.feature_extraction import FeatureHasher
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1)

del train

X_train = X_train.values.reshape(-1,28*28)
test = test.values.reshape(-1,28*28)
#Y_train = to_categorical(Y_train, num_classes = 10)

classes = (0,1,2,3,4,5,6,7,8,9)

def testMaxFeatures(X_train, Y_train, estimators):
    with open('RndForestMaxFeaturesResults.txt','a') as f:
        for i in [1,10,50,80,100,120,150]:
            print(i)
            clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators = estimators, max_features = i)
            scores = cross_val_score(clf, X_train, Y_train, cv=5)
            np.savetxt(f, scores, delimiter = ".")
            f.write('\n')
            for iter in scores:
                plt.plot(i, iter, 'bo')
            plt.plot(i, np.mean(scores), 'ro')
        clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators = 10)
        scores = cross_val_score(clf, X_train, Y_train, cv=5)
        f.write('\n')
        for iter in scores:
            plt.plot(0, iter, 'go')
        plt.plot(0, np.mean(scores), 'yo')
        plt.show()

def testMinLeaf(X_train, Y_train, maxFeatures, estimators):
    with open('RndForestMaxMinLeaf.txt','a') as f:
        for i in [1,10,50,80,100]:
            print(i)
            clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators = estimators, max_features = maxFeatures, min_samples_leaf = i)
            scores = cross_val_score(clf, X_train, Y_train, cv=5)
            np.savetxt(f, scores, delimiter = ".")
            f.write('\n')
            for iter in scores:
                plt.plot(i, iter, 'bo')
            plt.plot(i, np.mean(scores), 'ro')
        clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators = 300, max_features = 50)
        scores = cross_val_score(clf, X_train, Y_train, cv=5)
        f.write('\n')
        for iter in scores:
            plt.plot(0, iter, 'go')
        plt.plot(0, np.mean(scores), 'yo')
        plt.show()

#Function performing k-fold of the random forest methods on MNIST for several numbers of estimators
def crossValidation(X_train, Y_train):
    fig = plt.figure()

    with open('RndForestResults.txt','a') as f:
        for i in [70,75,80,85,90,95,100]:
            clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators = i)
            scores = cross_val_score(clf, X_train, Y_train, cv=5)
            np.savetxt(f, scores, delimiter = ".")
            f.write('\n')
            for iter in scores:
                plt.plot(i, iter, 'bo')
            plt.plot(i, np.mean(scores), 'ro')

    plt.xlabel("nb of estimators")
    plt.ylabel("Scores")
    plt.show()
    print(scores)

#Function calculation the confusion matrix for a given number of estimators
def randomForestComfusion(X_train, Y_train, estimators):
    clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators = estimators)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.9)
    clf.fit(X_train, Y_train)
    np.set_printoptions(suppress=True)
    results = clf.predict(X_test)
    matrix = confusion_matrix(Y_test, results, classes)
    matrix = matrix/matrix.sum(axis=0)[None,:]
    print(np.around(matrix, decimals = 4))


#Function generating the output file for kaggle submission. It receives the nb of estimators as a parameter
def mnistRandomForest(X_train, Y_train, estimators, maxFeatures):
    clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators = estimators, max_features = maxFeatures)
    clf.fit(X_train, Y_train)
    results = clf.predict(test)

    results = pd.Series(results,name="Label")

    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    submission.to_csv("cnn_mnist_datagen.csv",index=False)

#testMaxFeatures(X_train, Y_train, 300)
#testMinLeaf(X_train, Y_train, 300, 50)
#crossValidation(X_train, Y_train)
#randomForestComfusion(X_train, Y_train, 10)
mnistRandomForest(X_train, Y_train, 1000, 50)
