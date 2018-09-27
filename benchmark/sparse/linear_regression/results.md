# Linear Regression Benchmark Results 

## Summary
```
   Results below show the performance comparison of linear regression with MXNet vs Keras-Tensorflow using sparse tensors
```

### Configuration
| Dataset          | Synthetic(Randomly generated)                                |
| :--------------- | :----------------------------------------------------------- |
| Keras            | v2.2.2                                                      |
| TensorFlow       | v1.10.0                                                      |
| MXNet            | v1.3.0                                                      |


### Results 
##### Sparse data
###### Using 25 epochs
| Instance Type | GPUs  | Batch Size  | MXNet (Time/Epoch) | Keras-MXNet (Time/Epoch) | Keras-TensorFlow (Time/Epoch)  |
|-----|-----|-----|-----|-----|-----|
| C5.18XLarge |   0  | 16  | 52.89 sec | 334.32 sec | 237.28 sec |
| C5.18XLarge |   0  | 32 | 27.54 sec | 177.99 sec | 124.59 sec |
| C5.18XLarge |   0  | 64  | 13.78 sec | 85.22 sec | 60.86 sec |
| C5.18XLarge |   0  | 128  | 6.49 sec | 42.45 sec |  31.09 se |


### Note
For reproducing above results set seed to `7` by adding this line in the `run_sparse_benchmark` script - `np.random.seed(7)`
Run the file as `python run_sparse_benchmark.py`

### References
MXNet supports sparse data in 2 NDArray formats - CSRNDArray and RowSparseNDArray which are defined in `mxnet.ndarray.sparse` package
For further details on MXNet Sparse NDArray API check [documentation related to MXNet Sparse](https://mxnet.incubator.apache.org/api/python/ndarray/sparse.html)

Keras Input layer supports sparse data by setting a boolean placeholder value - check document for [Keras Input layer](https://keras.io/layers/core/#input)
