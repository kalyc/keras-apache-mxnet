"""
Linear Regression model with sparse synthetic data for Keras
"""

import time

from keras import Model
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras import backend as K
import mxnet as mx


def run_benchmark(train_data, train_label, eval_data, eval_label, batch_size, epochs):
    inputs = Input(shape=(10,), sparse=True)
    #print(K.is_sparse(inputs))
    #inputs = K.variable(train_data)
    predictions = Dense(10, activation='linear')(inputs)
    model = Model(outputs=predictions, inputs=inputs)

    if K.backend() == 'mxnet':
        sgd = SGD(lr=0.01, momentum=0.01, lazy_update=False)

    elif K.backend() == 'tensorflow':
        sgd = SGD(lr=0.01, momentum=0.01)
   
    model.compile(loss='mse', optimizer=sgd)

    start = time.time()
    model.fit(train_data, train_label,
              epochs=epochs,
              batch_size=batch_size)

    print("Keras-TF Sparse Benchmark Results")
    print("Batch Size")
    print(batch_size)
    print('Total Time')
    print(time.time() - start)

    mse = model.evaluate(eval_data, eval_label, verbose=0, batch_size=batch_size)

    print("Achieved {0:.6f} validation MSE".format(mse[0]))
    print(model.evaluate(eval_data, eval_label, verbose=1, batch_size=batch_size))
