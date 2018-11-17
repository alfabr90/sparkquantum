import math
import numpy as np

from sparkquantum.math.base import Base
from sparkquantum.math.vector import Vector, is_vector
from sparkquantum.utils.utils import Utils

__all__ = ['Matrix', 'is_matrix']


class Matrix(Base):
    """Class for matrices."""

    def __init__(self, rdd, shape, data_type=complex, coord_format=Utils.CoordinateDefault):
        """
        Build a `Matrix` object.

        Parameters
        ----------
        rdd : `RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a 2-dimensional tuple.
        data_type : type, optional
            The Python type of all values in this object. Default value is complex.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is `Utils.CoordinateDefault`.

        """
        super().__init__(rdd, shape, data_type=data_type)

        self._coordinate_format = coord_format

    @property
    def coordinate_format(self):
        return self._coordinate_format

    def dump(self, path):
        """Dump this object's RDD into disk.
        This method automatically converts the coordinate format to the default.

        Parameters
        ----------
        path : str
            The path where the dumped RDD will be located at

        """
        if self._coordinate_format == Utils.CoordinateMultiplier:
            rdd = self.data.map(
                lambda m: "{}, {}, {}".format(m[1][0], m[0], m[1][1])
            )
        elif self._coordinate_format == Utils.CoordinateMultiplicand:
            rdd = self.data.map(
                lambda m: "{}, {}, {}".format(m[0], m[1][0], m[1][1])
            )
        else:  # Utils.CoordinateDefault
            rdd = self.data.map(
                lambda m: " ".join([str(e) for e in m])
            )

        rdd.saveAsTextFile(path)

    def numpy_array(self):
        """Create a numpy array containing this object's RDD data.

        Returns
        -------
        ndarray
            The numpy array

        """
        data = self.data.collect()
        result = np.zeros(self._shape, dtype=self._data_type)

        if self._coordinate_format == Utils.CoordinateMultiplier:
            for e in data:
                result[e[1][0], e[0]] = e[1][1]
        elif self._coordinate_format == Utils.CoordinateMultiplicand:
            for e in data:
                result[e[0], e[1][0]] = e[1][1]
        else:  # Utils.CoordinateDefault
            for e in data:
                result[e[0], e[1]] = e[2]

        return result

    def _kron(self, other):
        other_shape = other.shape
        new_shape = (self._shape[0] * other_shape[0], self._shape[1] * other_shape[1])
        data_type = Utils.get_precendent_type(self._data_type, other.data_type)

        expected_elems = self._num_nonzero_elements * other.num_nonzero_elements
        expected_size = Utils.get_size_of_type(data_type) * expected_elems
        num_partitions = Utils.get_num_partitions(self.data.context, expected_size)

        rdd = self.data.map(
            lambda m: (0, m)
        ).join(
            other.data.map(
                lambda m: (0, m)
            ),
            numPartitions=num_partitions
        ).map(
            lambda m: (m[1][0], m[1][1])
        )

        if self._coordinate_format == Utils.CoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[0][1] * other_shape[1] + m[1][1], (m[0][0] * other_shape[0] + m[1][0], m[0][2] * m[1][2]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        elif self._coordinate_format == Utils.CoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0][0] * other_shape[0] + m[1][0], (m[0][1] * other_shape[1] + m[1][1], m[0][2] * m[1][2]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        else:  # Utils.CoordinateDefault
            rdd = rdd.map(
                lambda m: (m[0][0] * other_shape[0] + m[1][0], m[0][1] * other_shape[1] + m[1][1], m[0][2] * m[1][2])
            )

        return rdd, new_shape

    def kron(self, other, coord_format=Utils.CoordinateDefault):
        """Perform a tensor (Kronecker) product with another matrix.

        Parameters
        ----------
        other : `Matrix`
            The other matrix.
        coord_format : int, optional
            Indicate if the matrix must be returned in an apropriate format for multiplications.
            Default value is `Utils.CoordinateDefault`.

        Returns
        -------
        `Matrix`
            The resulting matrix.

        """
        if not is_matrix(other):
            if self._logger:
                self._logger.error("'Matrix' instance expected, not '{}'".format(type(other)))
            raise TypeError("'Matrix' instance expected, not '{}'".format(type(other)))

        rdd, new_shape = self._kron(other)

        return Matrix(rdd, new_shape, coord_format=coord_format)

    def norm(self):
        """Calculate the norm of this matrix.

        Returns
        -------
        float
            The norm of this matrix.

        """
        if self._coordinate_format == Utils.CoordinateMultiplier or self._coordinate_format == Utils.CoordinateMultiplicand:
            n = self.data.filter(
                lambda m: m[1][1] != complex()
            ).map(
                lambda m: m[1][1].real ** 2 + m[1][1].imag ** 2
            )
        else:  # Utils.CoordinateDefault
            n = self.data.filter(
                lambda m: m[2] != complex()
            ).map(
                lambda m: m[2].real ** 2 + m[2].imag ** 2
            )

        n = n.reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

    def is_unitary(self):
        """Check if this matrix is unitary by calculating its norm.

        Returns
        -------
        bool
            True if the norm of this matrix is 1.0, False otherwise.

        """
        round_precision = int(Utils.get_conf(self._spark_context, 'quantum.math.roundPrecision', default='10'))

        return round(self.norm(), round_precision) == 1.0

    def _multiply_matrix(self, other, coord_format):
        if self._shape[1] != other.shape[0]:
            if self._logger:
                self._logger.error("incompatible shapes {} and {}".format(self._shape, other.shape))
            raise ValueError("incompatible shapes {} and {}".format(self._shape, other.shape))

        shape = (self._shape[0], other.shape[1])
        num_partitions = max(self.data.getNumPartitions(), other.data.getNumPartitions())

        rdd = self.data.join(
            other.data, numPartitions=num_partitions
        ).map(
            lambda m: ((m[1][0][0], m[1][1][0]), m[1][0][1] * m[1][1][1])
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        )

        if coord_format == Utils.CoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[0][1], (m[0][0], m[1]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        elif coord_format == Utils.CoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0][0], (m[0][1], m[1]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        else:  # Utils.CoordinateDefault
            rdd = rdd.map(
                lambda m: (m[0][0], m[0][1], m[1])
            )

        return Matrix(rdd, shape, coord_format=coord_format)

    def _multiply_vector(self, other):
        if self._shape[1] != other.shape[0]:
            if self._logger:
                self._logger.error("incompatible shapes {} and {}".format(self._shape, other.shape))
            raise ValueError("incompatible shapes {} and {}".format(self._shape, other.shape))

        shape = other.shape

        rdd = self.data.join(
            other.data, numPartitions=self.data.getNumPartitions()
        ).map(
            lambda m: (m[1][0][0], m[1][0][1] * m[1][1])
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=self.data.getNumPartitions()
        )

        return Vector(rdd, shape)

    def multiply(self, other, coord_format=Utils.CoordinateDefault):
        """Multiply this matrix with another one or with a vector.

        Parameters
        ----------
        other `Matrix` or `Vector`
            A `Matrix` if multiplying another matrix, `Vector` otherwise.
        coord_format : int, optional
            Indicate if the matrix must be returned in an apropriate format for multiplications.
            Default value is `Utils.CoordinateDefault`. Not applicable when multiplying a `Vector`.

        Returns
        -------
        `Matrix` or `Vector`
            A `Matrix` if multiplying another matrix, `Vector` otherwise.

        Raises
        ------
        `TypeError`

        """
        if is_matrix(other):
            return self._multiply_matrix(other, coord_format)
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
