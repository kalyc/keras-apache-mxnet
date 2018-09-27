"""
Linear Regression model with sparse synthetic data for MXNet
"""

import time
import mxnet as mx


def run_benchmark(train_data, train_label, eval_data, eval_label, batch_size, epochs, start):
    train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, last_batch_handle='discard', label_name='label')

    eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size=batch_size,
                                  label_name='label', last_batch_handle='discard')

    X = mx.sym.Variable('data', stype='csr')
    weight = mx.symbol.Variable('weight', stype='row_sparse', shape=(train_data.shape[1], 1))
    pred = mx.sym.sparse.dot(X, weight)

    Y = mx.symbol.Variable('label')
    lro = mx.sym.LinearRegressionOutput(data=pred, label=Y, name='lro')

    # Create module
    model = mx.mod.Module(
        symbol=lro,
        data_names=['data'],
        label_names=['label']
    )

    model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    initializer = mx.initializer.Normal()
    model.init_params(initializer=initializer)
    sgd = mx.optimizer.SGD(learning_rate=0.1, momentum=0.9)
    model.init_optimizer(optimizer=sgd)

    # Use mean square error as the metric
    metric = mx.metric.create('MSE')

    for epoch in range(epochs):
        train_iter.reset()
        metric.reset()
        for batch in train_iter:
            model.forward(batch, is_train=True)  # compute predictions
            model.update_metric(metric, batch.label)  # accumulate prediction accuracy
            model.backward()  # compute gradients
            model.update()  # update parameters
        print("Epoch %d, Metric = %s" % (epoch, metric.get()))

    print("MXNet Sparse Benchmark Results")
    print("Dataset: Synthetic Sparse Data")
    print("Batch Size")
    print(batch_size)
    print("Total Time")
    print(time.time() - start)

    metric = mx.metric.MSE()
    mse = model.score(eval_iter, metric)
    print("Achieved {0:.6f} validation MSE".format(mse[0][1]))

    # Verify results on eval data
    print(model.score(eval_iter, metric))
