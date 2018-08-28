import pytest
from numpy.testing import assert_allclose
import numpy as np
import warnings
import scipy.sparse as sparse

from keras import backend as K


BACKEND = None

try:
    from keras.backend import mxnet_backend as KMX
    BACKEND = KMX
except ImportError:
    KMX = None
    warnings.warn('Could not import the MXNet backend')

class TestMXNetSparse(object):

    @pytest.mark.skipif((K.backend() != 'mxnet'),
                        reason='Testing only for MXNet backend')
    def test_is_sparse(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse_matrix = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        test_var = K.variable(x_sparse_matrix)

        assert K.is_sparse(test_var)

    @pytest.mark.skipif((K.backend() != 'mxnet'),
                        reason='Testing only for MXNet backend')
    def test_is_sparse_matrix(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse_matrix = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))

        assert K.is_sparse(x_sparse_matrix)

    @pytest.mark.skipif((K.backend() != 'mxnet'),
                        reason='Testing only for MXNet backend')
    def test_to_dense(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse_matrix = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        test_var = K.variable(x_sparse_matrix)

        assert_allclose(K.to_dense(test_var), x_sparse_matrix.toarray())

    @pytest.mark.skipif((K.backend() != 'mxnet'),
                        reason='Testing only for MXNet backend')
    def test_to_dense_matrix(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse_matrix = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))

        assert_allclose(K.to_dense(x_sparse_matrix), x_sparse_matrix.toarray())

    @pytest.mark.skipif((K.backend() != 'mxnet'),
                        reason='Testing only for MXNet backend')
    def test_sparse_sum(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        x_dense = x_sparse.toarray()
        test_var = K.variable(x_sparse)
        dense_var = K.variable(x_dense)

        k_s = K.eval(K.sum(test_var, axis=0))
        k_d = K.eval(K.sum(dense_var, axis=0))

        assert K.is_sparse(test_var)
        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    @pytest.mark.skipif((K.backend() != 'mxnet'),
                        reason='Testing only for MXNet backend')
    def test_sparse_mean(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        x_dense = x_sparse.toarray()
        test_var = K.variable(x_sparse)
        dense_var = K.variable(x_dense)

        k_s = K.eval(K.mean(test_var, axis=0))
        k_d = K.eval(K.mean(dense_var, axis=0))

        assert K.is_sparse(test_var)
        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    @pytest.mark.skipif((K.backend() != 'mxnet'),
                        reason='Testing only for MXNet backend')
    def test_sparse_mean_axis_none(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        x_dense = x_sparse.toarray()
        test_var = K.variable(x_sparse)
        dense_var = K.variable(x_dense)

        k_s = K.eval(K.mean(test_var))
        k_d = K.eval(K.mean(dense_var))

        assert K.is_sparse(test_var)
        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    @pytest.mark.skipif((K.backend() != 'mxnet'),
                        reason='Testing only for MXNet backend')
    def test_sparse_dot(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        x_dense = x_sparse.toarray()

        W = np.random.random((5, 4))

        t_W = K.variable(W)
        k_s = K.eval(K.dot(K.variable(x_sparse), t_W))
        k_d = K.eval(K.dot(K.variable(x_dense), t_W))

        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

if __name__ == '__main__':
    pytest.main([__file__])