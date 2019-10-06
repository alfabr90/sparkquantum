import math
import numpy as np

from sparkquantum.math.base import Base
from sparkquantum.math.vector import Vector, is_vector
from sparkquantum.utils.utils import Utils

__all__ = ['Matrix', 'is_matrix']


class Matrix(Base):
    """Class for matrices."""

    def __init__(self, rdd, shape, data_type=complex, coord_format=Utils.MatrixCoordinateDefault):
        """Build a :py:class:`sparkquantum.math.Matrix` object.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a two-dimensional tuple.
        data_type : type, optional
            The Python type of all values in this object. Default value is complex.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is :py:const:`Utils.MatrixCoordinateDefault`.

        """
        super().__init__(rdd, shape, data_type=data_type)

        self._coordinate_format = coord_format

    @property
    def coordinate_format(self):
        return self._coordinate_format

    def dump(self, path, glue=None, codec=None):
        """Dump this object's RDD to disk in many part-* files.

        Notes
        -----
        This method exports the data in the :py:const:`Utils.MatrixCoordinateDefault`.

        Parameters
        ----------
        path : str
            The path where the dumped RDD will be located at.
        glue : str, optional
            The glue string that connects each coordinate and value of each element in the RDD.
            Default value is None. In this case, it uses the 'quantum.dumpingGlue' configuration value.
        codec : str, optional
            Codec name used to compress the dumped data.
            Default value is None. In this case, it uses the 'quantum.dumpingCompressionCodec' configuration value.

        """
        if glue is None:
            glue = Utils.get_conf(self._spark_context, 'quantum.dumpingGlue')

        if codec is None:
            codec = Utils.get_conf(self._spark_context, 'quantum.dumpingCompressionCodec')

        if self._coordinate_format == Utils.MatrixCoordinateMultiplier:
            rdd = self.data.map(
                lambda m: glue.join((str(m[1][0]), str(m[0]), str(m[1][1])))
            )
        elif self._coordinate_format == Utils.MatrixCoordinateMultiplicand:
            rdd = self.data.map(
                lambda m: glue.join((str(m[0]), str(m[1][0]), str(m[1][1])))
            )
        else:  # Utils.MatrixCoordinateDefault
            rdd = self.data.map(
                lambda m: glue.join((str(m[0]), str(m[1]), str(m[2])))
            )

        rdd.saveAsTextFile(path, codec)

    def numpy_array(self):
        """Create a numpy array containing this object's RDD data.

        Returns
        -------
        ndarray
            The numpy array

        """
        data = self.data.collect()
        result = np.zeros(self._shape, dtype=self._data_type)

        if self._coordinate_format == Utils.MatrixCoordinateMultiplier:
            for e in data:
                result[e[1][0], e[0]] = e[1][1]
        elif self._coordinate_format == Utils.MatrixCoordinateMultiplicand:
            for e in data:
                result[e[0], e[1][0]] = e[1][1]
        else:  # Utils.MatrixCoordinateDefault
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

        if self._coordinate_format == Utils.MatrixCoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[0][1] * other_shape[1] + m[1][1], (m[0][0] * other_shape[0] + m[1][0], m[0][2] * m[1][2]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        elif self._coordinate_format == Utils.MatrixCoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0][0] * other_shape[0] + m[1][0], (m[0][1] * other_shape[1] + m[1][1], m[0][2] * m[1][2]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        else:  # Utils.MatrixCoordinateDefault
            rdd = rdd.map(
                lambda m: (m[0][0] * other_shape[0] + m[1][0], m[0][1] * other_shape[1] + m[1][1], m[0][2] * m[1][2])
            )

        return rdd, new_shape

    def kron(self, other, coord_format=Utils.MatrixCoordinateDefault):
        """Perform a tensor (Kronecker) product with another matrix.

        Parameters
        ----------
        other : :py:class:`sparkquantum.math.Matrix`
            The other matrix.
        coord_format : int, optional
            Indicate if the matrix must be returned in an apropriate format for multiplications.
            Default value is :py:const:`Utils.MatrixCoordinateDefault`.

        Returns
        -------
        :py:class:`sparkquantum.math.Matrix`
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
        if self._coordinate_format == Utils.MatrixCoordinateMultiplier or self._coordinate_format == Utils.MatrixCoordinateMultiplicand:
            n = self.data.filter(
                lambda m: m[1][1] != complex()
            ).map(
                lambda m: m[1][1].real ** 2 + m[1][1].imag ** 2
            )
        else:  # Utils.MatrixCoordinateDefault
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
        round_precision = int(Utils.get_conf(self._spark_context, 'quantum.math.roundPrecision'))

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

        if coord_format == Utils.MatrixCoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[0][1], (m[0][0], m[1]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        elif coord_format == Utils.MatrixCoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0][0], (m[0][1], m[1]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        else:  # Utils.MatrixCoordinateDefault
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

    def multiply(self, other, coord_format=Utils.MatrixCoordinateDefault):
        """Multiply this matrix with another one or with a vector.

        Parameters
        ----------
        other :py:class:`sparkquantum.math.Matrix` or :py:class:`sparkquantum.math.Vector`
            A :py:class:`sparkquantum.math.Matrix` if multiplying another matrix, :py:class:`sparkquantum.math.Vector` otherwise.
        coord_format : int, optional
            Indicate if the matrix must be returned in an apropriate format for multiplications.
            Default value is :py:const:`Utils.MatrixCoordinateDefault`. Not applicable when multiplying a :py:class:`sparkquantum.math.Vector`.

        Returns
        -------
        :py:class:`sparkquantum.math.Matrix` or :py:class:`sparkquantum.math.Vector`
            A :py:class:`sparkquantum.math.Matrix` if multiplying another matrix, :py:class:`sparkquantum.math.Vector` otherwise.

        Raises
        ------
        TypeError

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
    """Check whether argument is a :py:class:`sparkquantum.math.Matrix` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.math.Matrix` object, False otherwise.

    """
    return isinstance(obj, Matrix)
