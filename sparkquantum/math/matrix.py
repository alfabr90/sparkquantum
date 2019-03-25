import math
import numpy as np

from pyspark.sql import functions

from sparkquantum.math.base import Base
from sparkquantum.math.vector import Vector, is_vector
from sparkquantum.utils.utils import Utils

__all__ = ['Matrix', 'is_matrix']


class Matrix(Base):
    """Class for matrices."""

    def __init__(self, df, shape, data_type=complex):
        """Build a `Matrix` object.

        Parameters
        ----------
        df : `DataFrame`
            The base DataFrame of this object.
        shape : tuple
            The shape of this matrix object. Must be a 2-dimensional tuple.
        data_type : type, optional
            The Python type of all values in this object. Default value is complex.

        """
        super().__init__(df, shape, data_type=data_type)

    def dump(self, path, glue=None, codec=None):
        """Dump this object's DataFrame to disk in many part-* files.

        Parameters
        ----------
        path : str
            The path where the dumped DataFrame will be located at.
        glue : str, optional
            The glue string that connects each coordinate and value of each element in the DataFrame.
            Default value is `None`. In this case, it uses the 'quantum.dumpingGlue' configuration value.
        codec : str, optional
            Codec name used to compress the dumped data.
            Default value is `None`. In this case, it uses the 'quantum.dumpingCompressionCodec' configuration value.

        """
        if glue is None:
            glue = Utils.get_conf(self._spark_session, 'quantum.dumpingGlue')

        if codec is None:
            codec = Utils.get_conf(self._spark_session, 'quantum.dumpingCompressionCodec')

        df = self.data.map(
            lambda m: glue.join((str(m[0]), str(m[1]), str(m[2])))
        )

        df.saveAsTextFile(path, codec)

    def numpy_array(self):
        """Create a numpy array containing this object's DataFrame data.

        Returns
        -------
        ndarray
            The numpy array

        """
        data = self.data.collect()
        result = np.zeros(self._shape, dtype=self._data_type)

        for d in data:
            result[d['i'], d['j']] = d['v']

        return result

    def kron(self, other):
        """Perform a tensor (Kronecker) product with another matrix.

        Parameters
        ----------
        other : `Matrix`
            The other matrix.

        Returns
        -------
        `Matrix`
            The resulting matrix.

        """
        if not is_matrix(other):
            if self._logger:
                self._logger.error("'Matrix' instance expected, not '{}'".format(type(other)))
            raise TypeError("'Matrix' instance expected, not '{}'".format(type(other)))

        other_shape = other.shape
        new_shape = (self._shape[0] * other_shape[0], self._shape[1] * other_shape[1])
        data_type = Utils.get_precendent_type(self._data_type, other.data_type)

        a = self.data.select(
            functions.col('0').alias('a'), self.data['i'].alias('a_i'), self.data['j'].alias('a_j'), self.data['v'].alias('a_v')
        )

        b = other.data.select(
            functions.col('0').alias('b'), self.data['i'].alias('b_i'), self.data['j'].alias('b_j'), self.data['v'].alias('b_v')
        )

        df = a.join(
            b, a['a'] == b['b'], 'inner'
        ).select(
            a['a_i'] * other_shape[0] + b['b_i'], a['a_j'] * other_shape[1] + b['b_j'], a['a_v'] * b['b_v']
        )

        return Matrix(df, new_shape, data_type=data_type)

    def norm(self):
        """Calculate the norm of this matrix.

        Returns
        -------
        float
            The norm of this matrix.

        """
        n = self.data.filter(
            self.data['v'] != self._data_type()
        ).agg(
            functions.sum(functions.abs(self.data['v']) ** 2).alias('v')
        ).take(1)[0]['v']

        if n is None:
            return 0
        else:
            return math.sqrt(n)

    def is_unitary(self):
        """Check if this matrix is unitary by calculating its norm.

        Returns
        -------
        bool
            True if the norm of this matrix is 1.0, False otherwise.

        """
        round_precision = int(Utils.get_conf(self._spark_session, 'quantum.math.roundPrecision'))

        return round(self.norm(), round_precision) == 1.0

    def _multiply_matrix(self, other):
        if self._shape[1] != other.shape[0]:
            if self._logger:
                self._logger.error("incompatible shapes {} and {}".format(self._shape, other.shape))
            raise ValueError("incompatible shapes {} and {}".format(self._shape, other.shape))

        shape = (self._shape[0], other.shape[1])

        a = self.data.select(
            self.data['i'].alias('a_i'), self.data['j'].alias('a_j'), self.data['v'].alias('a_v'),
        )
        b = other.data.select(
            other.data['i'].alias('b_i'), other.data['j'].alias('b_j'), other.data['v'].alias('b_v'),
        )

        df = a.join(
            b, a['a_j'] == b['b_i'], 'inner'
        ).select(
            a['a_i'].alias('i'), b['b_j'].alias('j'), (a['a_v'] * b['b_v']).alias('v')
        ).groupBy(
            'i', 'j'
        ).agg(
            functions.sum('v').alias('v')
        )

        return Matrix(df, shape)

    def _multiply_vector(self, other):
        if self._shape[1] != other.shape[0]:
            if self._logger:
                self._logger.error("incompatible shapes {} and {}".format(self._shape, other.shape))
            raise ValueError("incompatible shapes {} and {}".format(self._shape, other.shape))

        shape = other.shape

        m = self.data.select(
            self.data['i'].alias('m_i'), self.data['j'].alias('m_j'), self.data['v'].alias('m_v'),
        )
        v = other.data.select(
            other.data['i'].alias('v_i'), other.data['v'].alias('v_v'),
        )

        df = m.join(
            v, m['m_j'] == v['v_i'], 'inner'
        ).select(
            m['m_i'].alias('i'), (m['m_v'] * v['v_v']).alias('v')
        ).groupBy(
            'i'
        ).agg(
            functions.sum('v').alias('v')
        )

        return Vector(df, shape)

    def multiply(self, other):
        """Multiply this matrix with another one or with a vector.

        Parameters
        ----------
        other `Matrix` or `Vector`
            A `Matrix` if multiplying another matrix, `Vector` otherwise.

        Returns
        -------
        `Matrix` or `Vector`
            A `Matrix` if multiplying another matrix, `Vector` otherwise.

        Raises
        ------
        `TypeError`

        """
        if is_matrix(other):
            return self._multiply_matrix(other)
        elif is_vector(other):
            return self._multiply_vector(other)
        else:
            if self._logger:
                self._logger.error("'Matrix' or 'Vector' instance expected, not '{}'".format(type(other)))
            raise TypeError("'Matrix' or 'Vector' instance expected, not '{}'".format(type(other)))


def is_matrix(obj):
    """Check whether argument is a `Matrix` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a `Matrix` object, False otherwise.

    """
    return isinstance(obj, Matrix)
