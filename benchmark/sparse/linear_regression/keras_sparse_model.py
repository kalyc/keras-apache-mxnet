"""
Linear Regression model with sparse synthetic data for Keras
"""
from __future__ import print_function

import time

from keras import Model
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras import backend as K


def run_benchmark(train_data, train_label, eval_data, eval_label, batch_size, epochs, start, is_sparse=True):
    if is_sparse:
        inputs = Input(batch_shape=(None, train_data.shape[1]), dtype='float32', sparse=True)
    else:
        inputs = Input(batch_shape=(None, train_data.shape[1]), dtype='float32', sparse=False)

    predictions = Dense(units=1, activation='linear', kernel_initializer='normal')(inputs)
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()

    sgd = SGD(lr=0.1, momentum=0.9)

    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

    model.fit(train_data,
              train_label,
              epochs=epochs,
              batch_size=batch_size, verbose=1)

    print("Keras Benchmark Results")
    if is_sparse:
        print("Dataset: Synthetic Sparse Data")
    else:
        print("Dataset: Synthetic Dense Data")
    print("Backend: ", K.backend().capitalize())
    print("Batch Size: ", batch_size)
    print("Total Time: ", time.time() - start)

    mse = model.evaluate(eval_data, eval_label, verbose=0, batch_size=batch_size)

    print("Achieved {0:.6f} validation MSE".format(mse[0]))
    print(model.evaluate(eval_data, eval_label, verbose=1, batch_size=batch_size))


