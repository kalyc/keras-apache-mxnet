import warnings

import mxnet as mx
import numpy as np
import pytest
import scipy.sparse as sparse
from keras import backend as K
from numpy.testing import assert_allclose

BACKEND = None

try:
    from keras.backend import mxnet_backend as KMX

    BACKEND = KMX
except ImportError:
    KMX = None
    warnings.warn('Could not import the MXNet backend')

pytestmark = pytest.mark.skipif(K.backend() != 'mxnet',
                                reason='Testing sparse support only for MXNet backend')


class TestMXNetSparse(object):
    def generate_test_sparse_matrix(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        return x_sparse

    def test_is_sparse(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()
        test_var = K.variable(test_sparse_matrix)

        assert K.is_sparse(test_var)

    def test_is_sparse_matrix(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()
        assert K.is_sparse(test_sparse_matrix)

    def test_to_dense(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()
        test_var = K.variable(test_sparse_matrix)

        assert_allclose(K.to_dense(test_var), test_sparse_matrix.toarray())

    def test_to_dense_matrix(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()

        assert_allclose(K.to_dense(test_sparse_matrix), test_sparse_matrix.toarray())

    def test_sparse_sum(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()
        test_dense_matrix = test_sparse_matrix.toarray()

        sparse_var = K.variable(test_sparse_matrix)
        dense_var = K.variable(test_dense_matrix)

        k_s = K.eval(K.sum(sparse_var, axis=0))
        k_d = K.eval(K.sum(dense_var, axis=0))

        assert K.is_sparse(sparse_var)
        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    def test_sparse_mean(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()
        test_dense_matrix = test_sparse_matrix.toarray()

        sparse_var = K.variable(test_sparse_matrix)
        dense_var = K.variable(test_dense_matrix)

        k_s = K.eval(K.mean(sparse_var, axis=0))
        k_d = K.eval(K.mean(dense_var, axis=0))

        assert K.is_sparse(sparse_var)
        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    def test_sparse_mean_axis_none(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()
        test_dense_matrix = test_sparse_matrix.toarray()

        sparse_var = K.variable(test_sparse_matrix)
        dense_var = K.variable(test_dense_matrix)

        k_s = K.eval(K.mean(sparse_var))
        k_d = K.eval(K.mean(dense_var))

        assert K.is_sparse(sparse_var)
        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    def test_sparse_dot(self):
        test_sparse_matrix = self.generate_test_sparse_matrix()
        test_dense_matrix = test_sparse_matrix.toarray()

        W = np.random.random((5, 4))

        t_W = K.variable(W)
        k_s = K.eval(K.dot(K.variable(test_sparse_matrix), t_W))
        k_d = K.eval(K.dot(K.variable(test_dense_matrix), t_W))

        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    def test_sparse_concat(self):
        test_sparse_matrix_1 = self.generate_test_sparse_matrix()
        test_sparse_matrix_2 = self.generate_test_sparse_matrix()

        assert K.is_sparse(K.variable(test_sparse_matrix_1))
        assert K.is_sparse(K.variable(test_sparse_matrix_2))

        test_dense_matrix_1 = test_sparse_matrix_1.toarray()
        test_dense_matrix_2 = test_sparse_matrix_2.toarray()

        k_s = K.concatenate(tensors=[K.variable(test_sparse_matrix_1), K.variable(test_sparse_matrix_2)], axis=0)
        k_s_d = K.eval(k_s)

        # mx.sym.sparse.concat only supported for axis=0
        k_d = K.eval(K.concatenate(tensors=[K.variable(test_dense_matrix_1), K.variable(test_dense_matrix_2)], axis=0))

        assert k_s_d.shape == k_d.shape
        assert_allclose(k_s_d, k_d, atol=1e-05)

    def test_sparse_concat_partial_dense(self):
        test_sparse_matrix_1 = self.generate_test_sparse_matrix()
        test_sparse_matrix_2 = self.generate_test_sparse_matrix()

        assert K.is_sparse(K.variable(test_sparse_matrix_1))
        assert K.is_sparse(K.variable(test_sparse_matrix_2))

        test_dense_matrix_1 = test_sparse_matrix_1.toarray()
        test_dense_matrix_2 = test_sparse_matrix_2.toarray()

        k_s = K.concatenate(tensors=[K.variable(test_sparse_matrix_1), K.variable(test_dense_matrix_2)], axis=0)
        k_s_d = K.eval(k_s)

        # mx.sym.sparse.concat only supported for axis=0
        k_d = K.eval(K.concatenate(tensors=[K.variable(test_dense_matrix_1), K.variable(test_dense_matrix_2)], axis=0))

        assert k_s_d.shape == k_d.shape
        assert_allclose(k_s_d, k_d, atol=1e-05)

    def test_sparse_concat_axis_non_zero(self):
        test_sparse_matrix_1 = self.generate_test_sparse_matrix()
        test_sparse_matrix_2 = self.generate_test_sparse_matrix()

        assert K.is_sparse(K.variable(test_sparse_matrix_1))
        test_dense_matrix_1 = test_sparse_matrix_1.toarray()
        test_dense_matrix_2 = test_sparse_matrix_2.toarray()

        k_s = K.concatenate(tensors=[K.variable(test_sparse_matrix_1), K.variable(test_dense_matrix_2)])
        k_s_d = K.eval(k_s)

        # mx.sym.sparse.concat only supported for axis=0
        k_d = K.eval(K.concatenate(tensors=[K.variable(test_dense_matrix_1), K.variable(test_dense_matrix_2)]))

        assert k_s_d.shape == k_d.shape
        assert_allclose(k_s_d, k_d, atol=1e-05)

    def _forward_pass(self, x):
        bind_values = K.dfs_get_bind_values(x)
        executor = x.symbol.simple_bind(mx.cpu(), grad_req='null')
        for v in executor.arg_dict:
            bind_values[v].copyto(executor.arg_dict[v])
        outputs = executor.forward(is_train=K.learning_phase())
        return outputs

    def test_sparse_embedding(self):
        # Sparse data
        sparse_matrix = self.generate_test_sparse_matrix()
        test_sparse_data = K.variable(sparse_matrix)

        x_d = np.array([0, 7, 1, 4], dtype=np.float32)
        x_r = np.array([0, 2, 1, 3], dtype=np.int64)
        x_c = np.array([0, 1, 2, 3], dtype=np.int64)

        sparse_weight = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        test_sparse_weight = K.variable(sparse_weight)
        assert K.is_sparse(sparse_weight)
        assert K.is_sparse(test_sparse_data)

        # Dense data
        dense_matrix = sparse_matrix.toarray()
        test_dense_data = K.variable(dense_matrix)

        dense_weight = sparse_weight.toarray()
        test_dense_weight = K.variable(dense_weight)

        k_S = K.embedding(test_sparse_data, test_sparse_weight, 4, 5, sparse_grad=True)
        k_D = K.embedding(test_dense_data, test_dense_weight, 4, 5)

        assert k_S.shape == k_D.shape

        x = self._forward_pass(k_S)
        y = self._forward_pass(k_D)
        assert x.sort() == y.sort()

if __name__ == '__main__':
    pytest.main([__file__])
