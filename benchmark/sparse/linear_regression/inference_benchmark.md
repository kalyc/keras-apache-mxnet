# Linear Regression Benchmark Results 

## Summary
```
   Results below show the performance comparison of linear regression with MXNet vs Keras-Tensorflow using sparse tensors
```                                                   

### Results
### Inference Benchmark
### Configuration
| Dataset          | Synthetic(Randomly generated)                                |
| :--------------- | :----------------------------------------------------------- |
| Keras            | v2.2.2                                                      |
| TensorFlow       | v1.11.0                                                     |
| MXNet-mkl         | v1.3.0   

###### Using 10 epochs
#### CPU
##### Speed
###### Note speed calculated using an average of 5 runs
| Instance Type | GPUs  | Batch Size  | Keras-MXNet (Time/Epoch) | Keras-TensorFlow (Time/Epoch)  |
|-----|-----|-----|-----|-----|
| C5.8XLarge |   0  | 64  | 64.2 sec | 64.3 sec
| C5.8XLarge |   0  | 128 | 32.1 sec | 31.4 sec 
| C5.8XLarge |   0  | 256 | 16.5 sec | 16.3 sec
| C5.8XLarge |   0  | 512 | 8.3 sec | 8.8 sec 
| C5.8XLarge |   0  | 1024 | 4.3 sec | 4.4 sec

#### Memory utilization
| Instance Type | GPUs  | Batch Size | Keras-MXNet (Mem %) | Keras-TensorFlow (Mem %)  |
|-----|-----|-----|-----|-----|
| C5.8XLarge |   0  | 64  | 0.4 | 0.5 |
| C5.8XLarge |   0  | 128 | 0.4 | 0.5 | 
| C5.8XLarge |   0  | 256 | 0.4 | 0.5  |
| C5.8XLarge |   0  | 512 | 0.4 | 0.5 |
| C5.8XLarge |   0  | 1024 | 0.4 | 0.5 |

#### Memory consumed
| Instance Type | GPUs  | Batch Size | Keras-MXNet (MB) | Keras-TensorFlow (MB)  |
|-----|-----|-----|-----|-----|
| C5.8XLarge |   0  | 64  | 1630.8 | 1573.8 |
| C5.8XLarge |   0  | 128 | 1574.7 | 1561.2 | 
| C5.8XLarge |   0  | 256 | 1477.8 | 1501.4  |
| C5.8XLarge |   0  | 512 | 1407.0| 1472.5 |
| C5.8XLarge |   0  | 1024 | 1336.3 | 1466.8 |

#### GPU
### Configuration
| Dataset          | Synthetic(Randomly generated)                                |
| :--------------- | :----------------------------------------------------------- |
| Keras            | v2.2.2                                                      |
| TensorFlow-GPU   | v1.11.0                                                      |
| MXNet-cu90mkl    | v1.3.0   

###### Using 10 epochs
##### Single GPU
##### Speed
###### Note speed calculated using an average of 5 runs
| Instance Type | GPUs  | Batch Size  | Keras-MXNet (Time/Epoch) | Keras-TensorFlow (Time/Epoch)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   1  | 64  | 3.0 sec | 2.0 sec
| P3.8XLarge |   1  | 128 | 2.3 sec | 1.2 sec 
| P3.8XLarge |   1  | 256 | 1.2 sec | 0.7 sec
| P3.8XLarge |   1  | 512 | 0.8 sec | 0.5 sec
| P3.8XLarge |   1  | 1024 | 0.4 sec | 0.4 sec

##### Memory utilization
| Instance Type | GPUs  | Batch Size | Keras-MXNet (GPU Utilization %) | Keras-TensorFlow (GPU Utilization %)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   1  | 64  | 9 | 6
| P3.8XLarge |   1  | 128 | 8 | 7
| P3.8XLarge |   1  | 256 | 8 | 7
| P3.8XLarge |   1  | 512 | 8 | 7
| P3.8XLarge |   1  | 1024 | 7 | 8

##### Memory consumed
| Instance Type | GPUs  | Batch Size | Keras-MXNet (MB) | Keras-TensorFlow (MB)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   1  | 64  | 966.7 | 16135.5
| P3.8XLarge |   1  | 128 | 970.9 | 16135.5
| P3.8XLarge |   1  | 256 | 973.1 | 16135.5
| P3.8XLarge |   1  | 512 | 994.1 | 16135.5
| P3.8XLarge |   1  | 1024 | 987.7 | 16135.5

##### Multi-GPU
##### Speed
###### Benchmark results on multi GPU calculated using an average of 5 runs
| Instance Type | GPUs  | Batch Size | Keras-MXNet (Time/Epoch) | Keras-TensorFlow (Time/Epoch)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   2  | 512 | 0.86 sec | Not supported
| P3.8XLarge |   4  | 1024  | 0.71 sec | Not supported

##### Memory utilization
| Instance Type | GPUs  | Batch Size | Keras-MXNet (GPU Utilization %) | Keras-TensorFlow (GPU Utilization %)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |  2  | 512 | 5.5  | Not supported
| P3.8XLarge |  4  | 1024 | 1.5 | Not supported

##### Memory consumed
| Instance Type | GPUs  | Batch Size | Keras-MXNet (MB) | Keras-TensorFlow (MB)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   2  | 512 | 972.0 | Not supported
| P3.8XLarge |   4 | 1024 | 969.9 | Not supported

### Note
For reproducing above results start time before invoking model.predict()
Run the file as `python run_sparse_benchmark.py`

### References
MXNet supports sparse data in 2 NDArray formats - CSRNDArray and RowSparseNDArray which are defined in `mxnet.ndarray.sparse` package
For further details on MXNet Sparse NDArray API check [documentation related to MXNet Sparse](https://mxnet.incubator.apache.org/api/python/ndarray/sparse.html)

Keras Input layer supports sparse data by setting a boolean placeholder value - check document for [Keras Input layer](https://keras.io/layers/core/#input)