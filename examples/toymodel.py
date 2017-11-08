from kerasgym import GymModel

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import numpy as np


class Model(GymModel):

    def model(self):
        """2-8-8-1 fully connected model:
        two inputs, two hidden layers and one sigmoid output"""
        model = Sequential()
        model.add(Dense(3, activation='relu', input_shape=(2,)))
        model.add(Dense(1, activation='tanh'))

        optimizer = SGD(lr=0.05)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def train(self, model, epochs=10, initial_epoch=0):
        """Train our model to perform the XOR operation"""
        n = 1024
        X_train = np.random.choice(2, (n, 2))
        Y_train = np.logical_xor(X_train[:, 0], X_train[:, 1]).astype(int)
        print X_train, Y_train
        history = model.fit(X_train, Y_train,
                            validation_split=0.25,
                            batch_size=16, epochs=epochs + initial_epoch,
                            initial_epoch=initial_epoch,
                            verbose=1)

        return history
