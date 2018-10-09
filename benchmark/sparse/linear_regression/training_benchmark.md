# Linear Regression Benchmark Results 

## Summary
```
   Results below show the performance comparison of linear regression with MXNet vs Keras-Tensorflow using sparse tensors
```                                                   

### Results
### Training Benchmark
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
| C5.8XLarge |   0  | 64  | 715.4 sec | 676.6 sec
| C5.8XLarge |   0  | 128 | 346.2 sec | 341.1 sec 
| C5.8XLarge |   0  | 256 | 168.9 sec | 165.5
| C5.8XLarge |   0  | 512 | 89.3 sec | 83.8 sec 
| C5.8XLarge |   0  | 1024 | 50.5 sec | 48.6 sec

#### Memory consumed
| Instance Type | GPUs  | Batch Size | Keras-MXNet (%) | Keras-TensorFlow (%)  |
|-----|-----|-----|-----|-----|
| C5.8XLarge |   0  | 64  | 0.4 | 0.4 |
| C5.8XLarge |   0  | 128 | 0.4 | 0.4 | 
| C5.8XLarge |   0  | 256 | 0.3 | 0.4  |
| C5.8XLarge |   0  | 512 | 0.3 | 0.4 |
| C5.8XLarge |   0  | 1024 | 0.3 | 0.4 |

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
| P3.8XLarge |   1  | 64  | 62.3 sec | 59.7 sec
| P3.8XLarge |   1  | 128 | 34.8 sec | 37.7 sec 
| P3.8XLarge |   1  | 256 | 19.3 sec | 28.4 sec
| P3.8XLarge |   1  | 512 | 19.3 sec | 28.4 sec
| P3.8XLarge |   1  | 1024 | 19.3 sec | 28.4 sec

##### Memory consumed
| Instance Type | GPUs  | Batch Size | Keras-MXNet (%) | Keras-TensorFlow (%)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   1  | 64  | 19 | 14
| P3.8XLarge |   1  | 128 | 19 | 16
| P3.8XLarge |   1  | 256 | 20 | 14

##### Multi-GPU
##### Speed
###### Benchmark results on multi GPU calculated using an average of 5 runs
| Instance Type | GPUs  | Batch Size | Keras-MXNet (Time/Epoch) | Keras-TensorFlow (Time/Epoch)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   2  | 512 | 16 sec | Not supported
| P3.8XLarge |   4  | 1024  | 11 sec | Not supported

##### Memory consumed
| Instance Type | GPUs  | Batch Size | Keras-MXNet (%) | Keras-TensorFlow (%)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |  2  | 512 | 12  | Not supported
| P3.8XLarge |  4  | 1024 | 9 | Not supported

### Note
For reproducing above results start time before invoking model.fit()
Run the file as `python run_sparse_benchmark.py`

### References
MXNet supports sparse data in 2 NDArray formats - CSRNDArray and RowSparseNDArray which are defined in `mxnet.ndarray.sparse` package
For further details on MXNet Sparse NDArray API check [documentation related to MXNet Sparse](https://mxnet.incubator.apache.org/api/python/ndarray/sparse.html)

Keras Input layer supports sparse data by setting a boolean placeholder value - check document for [Keras Input layer](https://keras.io/layers/core/#input)
