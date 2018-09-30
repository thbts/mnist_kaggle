import pandas as pd
import numpy as np
import keras.backend as K

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,\
                         BatchNormalization, Input, concatenate, \
                         GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from keras.applications import ResNet50
from sklearn.model_selection import train_test_split

np.random.seed(2)

def dense_block(filters, inputs):
    """Dense block definition"""
    cnn_1 = Conv2D(filters = filters, kernel_size = (3, 3),padding = 'Same',
                     activation ='relu')(inputs)
    cnn_1 = Dropout(0.2)(cnn_1)
    cnn_1 = BatchNormalization()(cnn_1)
    cnn_2 = Conv2D(filters = filters, kernel_size = (3, 3),padding = 'Same',
                     activation ='relu')(cnn_1)
    cnn_2 = Dropout(0.2)(cnn_2)
    cnn_2 = BatchNormalization()(cnn_2)
    inp_cnn_3 = concatenate([cnn_1,cnn_2])
    cnn_3 = Conv2D(filters = filters, kernel_size = (3, 3),padding = 'Same',
                     activation ='relu')(inp_cnn_3)
    cnn_3 = Dropout(0.2)(cnn_3)
    cnn_3 = BatchNormalization()(cnn_3)
    inp_cnn_4 = concatenate([cnn_1,cnn_2,cnn_3])
    cnn_4 = Conv2D(filters = filters, kernel_size = (3, 3),padding = 'Same',
                     activation ='relu')(inp_cnn_4)
    cnn_4 = Dropout(0.2)(cnn_4)
    cnn_4 = BatchNormalization()(cnn_4)
    cnn_5 = Conv2D(filters = filters, kernel_size = (3, 3),padding = 'Same',
                     activation ='relu')(cnn_4)
    cnn_5 = Dropout(0.2)(cnn_5)
    output = BatchNormalization()(cnn_5)
    return output

class ResNet_mnist:
    """Class for ResNet50 adapted to mnist"""

    def __init__(self, model=None):
        if model is not None:
            self.model = Model.load(model)
        else:
            self.model = self.define_model()

    def define_model(self, trained=True):
        """Build and return a ResNet50 with the right architecture.
        ----------
        trained: Boolean
            Set to true load imagenet weights.
        Returns
        -------
        model: instance of Keras.Model
        ---------
        """
        base_model = ResNet50(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def train(self, X_train, X_val, Y_train, Y_val):
        """Train the network for mnist.
        ----------
        X_train: array
            Training dataset. Size (N, 28, 28, 1), N being fixed by the split.
        Y_train: array
            Training target. Size (N, 10)
        X_val: array
            Validation dataset. Size (N, 28, 28, 1), N being fixed by the split.
        Y_val: array
            Validation target. Size (N, 10)
        Returns
        -------
        ---------
        """
        #Reshaping the input data so it fits with the 3 channels input of ResNet
        X_train_3_channels = np.squeeze(np.stack((X_train,)*3, -1))
        X_val_3_channels = np.squeeze(np.stack((X_val,)*3, -1))

        # first: train only the top layers (which were randomly initialized)
        epochs = 1
        batch_size = 128
        for layer in self.model.layers[:-2]:
            layer.trainable = False
        self.model.compile(optimizer='Nadam', loss='categorical_crossentropy',  metrics=['accuracy'])
        # train the model on the new data for a few epochs
        self.model.fit(X_train_3_channels,Y_train, batch_size,epochs=epochs,validation_data=(X_val_3_channels,Y_val))

        # At this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers.
        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        epochs = 10
        batch_size = 64
        for layer in self.model.layers:
            layer.trainable = True
        # from keras.optimizers import SGD
        # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train_3_channels,Y_train, batch_size,epochs=epochs,validation_data=(X_val_3_channels,Y_val))

    def test(self, test, submission=False):
        """Test the model, save it and produce the submission file if needed. If
        not, return the results for ensembling purposes."""
        # predict results
        results = self.model.predict(test)
        self.model.save('resnet_fine_tuned.h5')
        if submission!=True:
            return results
        # select the index with the maximum probability
        results = np.argmax(results,axis = 1)
        results = pd.Series(results,name="Label")
        submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
        submission.to_csv("cnn_mnist_resnet50.csv",index=False)

class DenseNet_mnist:
    """Class for DenseNet adapted to mnist"""

    def __init__(self, model=None):
        if model is not None:
            self.model = Model.load(model)
        else:
            self.model = self.define_model()

    def define_model(self):
        """This creates a model that includes the Input layer and three Dense layers"""
        inputs = Input(shape=(28,28,1,))
        # First dense block
        cnn1_5 = dense_block(32, inputs)
        inp2_1 = MaxPool2D(pool_size=(2,2))(cnn1_5)
        ##Second dense block
        cnn2_5 = dense_block(64, inp2_1)
        inp3_1 = MaxPool2D(pool_size=(2,2))(cnn2_5)
        ##Third dense block
        cnn3_5 = dense_block(128, inp3_1)
        vector = Flatten()(cnn3_5)
        vector = Dense(256, activation = "relu")(vector)
        vector = BatchNormalization()(vector)
        vector = Dropout(0.5)(vector)
        predictions = Dense(10, activation = "softmax")(vector)
        model = Model(inputs=inputs, outputs=predictions)
        return model

    # def train(self, X_train, X_val, Y_train, Y_val):
    def train(self, X_train, Y_train):
        """Train the network for mnist.
        ----------
        X_train: array
            Training dataset. Size (N, 28, 28, 1), N being fixed by the split.
        Y_train: array
            Training target. Size (N, 10)
        X_val: array
            Validation dataset. Size (N, 28, 28, 1), N being fixed by the split.
        Y_val: array
            Validation target. Size (N, 10)
        Returns
        -------
        ---------
        """
        self.model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])
        epochs = 10
        batch_size = 128

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.9, 1.1],
            rescale=1./255,
            validation_split=0.1
            )

        datagen.fit(X_train)

        # fits the model on batches with real-time data augmentation:
        self.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                 steps_per_epoch=len(X_train) / batch_size, epochs=epochs)

        # self.model.fit(X_train,Y_train, batch_size,epochs=epochs,validation_data=(X_val,Y_val))

    def test(self, test, name_file, submission=False):
        """Test the model, save it and produce the submission file if needed. If
        not, return the results for ensembling purposes."""
        # predict results
        results = self.model.predict(test)
        self.model.save(name_file+'.h5')
        if submission!=True:
            return results
        # select the indix with the maximum probability
        results = np.argmax(results,axis = 1)
        results = pd.Series(results,name="Label")
        submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
        submission.to_csv(name_file+'.csv',index=False)
        return results


class Expert_Net17_mnist:
    """Class for an expert network to differentiate 1 and 7 adapted to mnist"""

    def __init__(self, model=None):
        if model is not None:
            self.model = Model.load(model)
        else:
            self.model = self.define_model()

    def define_model(self):
        #TODO change architecture
        model = Sequential()
        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                         activation ='relu', input_shape = (28,28,1)))
        model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                         activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                         activation ='relu'))
        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                         activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation = "softmax"))
        return model

    def train(self, X_train, Y_train):
        """Train the network for mnist.
        ----------
        X_train: array
            Training dataset. Size (N, 28, 28, 1), N being the entire dataset size.
        Y_train: array
            Training target. Size (N, 10)
        Returns
        -------
        ---------
        """
        samples_per_class = len(Y_train)/10 #in order to avoid imbalance between 1, 7 and the rest of the classes
        dataset_size = samples_per_class*3
        X_train_17 = []
        Y_train_17 = []
        done = False
        for x, y in zip(X_train, Y_train):
            if done:
                break
            done = True
            if y[0] or y[6]:
                X_train_17.append(x)
                Y_train_17.append([y[0], y[6], 0])
            elif samples_per_class > 0:
                X_train_17.append(x)
                Y_train_17.append([0, 0, 1])
                samples_per_class -= 1
            if dataset_size > len(Y_train_17):
                done = False
        X_train_17 = np.asarray(X_train_17)
        Y_train_17 = np.asarray(Y_train_17)
        X_train_17, X_val_17, Y_train_17, Y_val_17 = train_test_split(X_train_17, Y_train_17, test_size = 0.1)

        epochs = 5
        batch_size = 64
        self.model.compile(optimizer = 'Nadam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        self.model.fit(X_train_17,Y_train_17, batch_size,epochs=epochs,validation_data=(X_val_17,Y_val_17))

    def test(self, test):
        """Test the model, save it and return the results for ensembling purposes."""
        # predict results
        results = self.model.predict(test)
        self.model.save('expert_17.h5')
        return results
