import math

import numpy as np
from pyspark import RDD

from sparkquantum.math.base import Base
from sparkquantum.utils.utils import Utils

__all__ = ['Matrix', 'is_matrix']


class Matrix(Base):
    """Class for matrices."""

    def __init__(self, rdd, shape, data_type=complex,
                 coordinate_format=Utils.MatrixCoordinateDefault, num_elements=None):
        """Build a matrix object.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this object. Must be a two-dimensional tuple.
        data_type : type, optional
            The Python type of all values in this object. Default value is complex.
        coordinate_format : int, optional
            The coordinate format of this object. Default value is :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault`.
        num_elements : int, optional
            The expected (or definitive) number of elements. This helps to find a
            better number of partitions when (re)partitioning the RDD. Default value is None.

        """
        super().__init__(rdd, num_elements=num_elements)

        self._shape = shape
        self._data_type = data_type
        self._coordinate_format = coordinate_format

        self._size = self._shape[0] * self._shape[1]

        if not Utils.is_shape(shape):
            self._logger.error("invalid shape")
            raise ValueError("invalid shape")

    @property
    def shape(self):
        """tuple"""
        return self._shape

    @property
    def data_type(self):
        """type"""
        return self._data_type

    @property
    def coordinate_format(self):
        """int"""
        return self._coordinate_format

    @property
    def size(self):
        """int"""
        return self._size

    def __str__(self):
        return '{} with shape {}'.format(self.__class__.__name__, self._shape)

    def sparsity(self):
        """Calculate the sparsity of this matrix.

        Returns
        -------
        float
            The sparsity of this matrix.

        """
        return 1.0 - self._num_elements / self._size

    def dump(self, path, glue=None, codec=None, filename=None):
        """Dump this object's RDD to disk in a unique file or in many part-* files.

        Notes
        -----
        Depending on the chosen dumping mode, this method calls the RDD's :py:func:`pyspark.RDD.collect` method.
        This is not suitable for large working sets, as all data may not fit into driver's main memory.
        This method exports the data in the :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault` format.

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
        filename : str, optional
            File name used when the dumping mode is in a single file. Default value is None.
            In this case, a temporary named file is generated inside the informed path.

        Raises
        ------
        ValueError
            If the chosen 'quantum.math.dumpingMode' configuration is not valid.

        """
        if glue is None:
            glue = Utils.get_conf(self._spark_context, 'quantum.dumpingGlue')

        if codec is None:
            codec = Utils.get_conf(
                self._spark_context,
                'quantum.dumpingCompressionCodec')

        dumping_mode = int(
            Utils.get_conf(
                self._spark_context,
                'quantum.math.dumpingMode'))

        rdd = Utils.remove_zeros(
            Utils.change_coordinate(
                self._data,
                self._coordinate_format,
                Utils.MatrixCoordinateDefault),
            self._data_type,
            Utils.MatrixCoordinateDefault)

        rdd = rdd.map(
            lambda m: glue.join((str(m[0]), str(m[1]), str(m[2])))
        )

        if dumping_mode == Utils.DumpingModeUniqueFile:
            data = rdd.collect()

            Utils.create_dir(path)

            if not filename:
                filename = Utils.get_temp_path(path)
            else:
                filename = Utils.append_slash_dir(path) + filename

            if len(data):
                with open(filename, 'a') as f:
                    for d in data:
                        f.write(d + "\n")
        elif dumping_mode == Utils.DumpingModePartFiles:
            rdd.saveAsTextFile(path, codec)
        else:
            self._logger.error("invalid dumping mode")
            raise ValueError("invalid dumping mode")

    def numpy_array(self):
        """Create a numpy array containing this object's RDD data.

        Notes
        -----
        This method calls the :py:func:`pyspark.RDD.collect` method. This is not suitable for large working sets,
        as all data may not fit into main memory.

        Returns
        -------
        :py:class:`numpy.ndarray`
            The numpy array.

        """
        rdd = Utils.remove_zeros(
            Utils.change_coordinate(
                self._data,
                self._coordinate_format,
                Utils.MatrixCoordinateDefault),
            self._data_type,
            Utils.MatrixCoordinateDefault)

        data = rdd.collect()

        result = np.zeros(self._shape, dtype=self._data_type)

        for e in data:
            result[e[0], e[1]] = e[2]

        return result

    def change_coordinate(self, coordinate_format):
        """Change the coordinate format of this object.

        Parameters
        ----------
        coordinate_format : int
            The new coordinate format of this object.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            A reference to this object.

        """
        self._data = Utils.change_coordinate(
            self._data,
            old_coordinate=self._coordinate_format,
            new_coordinate=coordinate_format)

        self._coordinate_format = coordinate_format

        return self

    def _transpose(self):
        rdd = Utils.remove_zeros(
            Utils.change_coordinate(
                self._data,
                self._coordinate_format,
                Utils.MatrixCoordinateDefault),
            self._data_type,
            Utils.MatrixCoordinateDefault)

        shape = (self._shape[1], self._shape[0])

        rdd = rdd.map(
            lambda m: (m[1], m[0], m[2])
        )

        return rdd, shape

    def transpose(self):
        """Transpose this matrix.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            The resulting matrix.

        """
        rdd, shape = self._transpose()

        return Matrix(rdd, shape, data_type=self._data_type,
                      coordinate_format=Utils.MatrixCoordinateDefault)

    def _kron(self, other):
        other_shape = other.shape
        new_shape = (
            self._shape[0] *
            other_shape[0],
            self._shape[1] *
            other_shape[1])

        data_type = Utils.get_precedent_type(self._data_type, other.data_type)

        rdd = self._data
        other_rdd = other.data

        # TODO: improve
        if self._num_elements is not None and other.num_elements is not None:
            expected_elements = self._num_elements * other.num_elements

            num_partitions = Utils.get_num_partitions(
                self._spark_context,
                Utils.get_size_of_type(data_type) * expected_elements
            )
        else:
            expected_elements = None

            num_partitions = max(
                rdd.getNumPartitions(),
                other_rdd.getNumPartitions())

        rdd = rdd.map(
            lambda m: (0, m)
        ).join(
            other_rdd.map(
                lambda m: (0, m)
            ),
            numPartitions=num_partitions
        ).map(
            lambda m: (m[1][0], m[1][1])
        ).map(
            lambda m: (
                m[0][0] * other_shape[0] + m[1][0],
                m[0][1] * other_shape[1] + m[1][1],
                m[0][2] * m[1][2])
        )

        rdd = Utils.remove_zeros(rdd, data_type, Utils.MatrixCoordinateDefault)

        return rdd, new_shape, data_type, expected_elements

    def kron(self, other):
        """Perform a tensor (Kronecker) product with another matrix.

        Parameters
        ----------
        other : :py:class:`sparkquantum.math.matrix.Matrix`
            The other matrix.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            The resulting matrix.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.math.matrix.Matrix`.

        """
        if not is_matrix(other):
            self._logger.error(
                "'Matrix' instance expected, not '{}'".format(
                    type(other)))
            raise TypeError(
                "'Matrix' instance expected, not '{}'".format(
                    type(other)))

        rdd, shape, data_type, num_elements = self._kron(other)

        return Matrix(rdd, shape, data_type=data_type,
                      num_elements=num_elements)

    def norm(self):
        """Calculate the norm of this matrix.

        Returns
        -------
        float
            The norm of this matrix.

        """
        rdd = Utils.remove_zeros(
            Utils.change_coordinate(
                self._data,
                self._coordinate_format,
                Utils.MatrixCoordinateDefault),
            self._data_type,
            Utils.MatrixCoordinateDefault)

        if self._data_type == complex:
            def __map(m):
                return m[2].real ** 2 + m[2].imag ** 2
        else:
            def __map(m):
                return m[2] ** 2

        n = rdd.map(
            __map
        ).reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

    def is_unitary(self):
        """Check if this matrix is unitary by calculating its norm.

        Notes
        -----
        This method uses the 'quantum.math.roundPrecision' configuration to round the calculated norm.

        Returns
        -------
        bool
            True if the norm of this matrix is 1.0, False otherwise.

        """
        round_precision = int(
            Utils.get_conf(
                self._spark_context,
                'quantum.math.roundPrecision'))

        return round(self.norm(), round_precision) == 1.0

    def _sum_matrix(self, other, constant):
        if self._shape[0] != other.shape[0] or self._shape[1] != other.shape[1]:
            self._logger.error(
                "incompatible shapes {} and {}".format(
                    self._shape, other.shape))
            raise ValueError(
                "incompatible shapes {} and {}".format(
                    self._shape, other.shape))

        data_type = Utils.get_precedent_type(
            self._data_type, other.data_type)

        rdd = self._data
        other_rdd = other.data

        if self._num_elements is not None and other.num_elements is not None:
            expected_elements = self._num_elements + other.num_elements

            num_partitions = Utils.get_num_partitions(
                self._spark_context,
                Utils.get_size_of_type(data_type) * expected_elements
            )
        else:
            expected_elements = None

            num_partitions = max(
                rdd.getNumPartitions(),
                other_rdd.getNumPartitions())

        def __map(m):
            a = m[1][0]
            b = m[1][1]

            if a is None:
                a = 0

            if b is None:
                b = 0

            return (m[0][0], m[0][1], a + constant * b)

        rdd = rdd.fullOuterJoin(
            other_rdd, numPartitions=num_partitions
        ).map(
            __map
        )

        rdd = Utils.remove_zeros(rdd, data_type, Utils.MatrixCoordinateDefault)

        return rdd, self._shape, data_type, expected_elements

    def _sum_scalar(self, other, constant):
        if other == type(other)():
            return self._data, self._data_type

        data_type = Utils.get_precedent_type(
            self._data_type, type(other))

        rdd = self._data

        other_rdd = self._spark_context.range(
            self._shape[0]
        ).cartesian(
            self._spark_context.range(self._shape[1])
        ).map(
            lambda m: (m, constant * other)
        )

        expected_elements = self._size

        num_partitions = Utils.get_num_partitions(
            self._spark_context,
            Utils.get_size_of_type(data_type) * expected_elements
        )

        def __map(m):
            a = m[1][0]
            b = m[1][1]

            if a is None:
                a = 0

            return (m[0][0], m[0][1], a + constant * b)

        rdd = rdd.rightOuterJoin(
            other_rdd, num_partitions=num_partitions
        ).map(
            __map
        )

        rdd = Utils.remove_zeros(rdd, data_type, Utils.MatrixCoordinateDefault)

        return rdd, self._shape, data_type, expected_elements

    def _sum(self, other, constant):
        if is_matrix(other):
            return self._sum_matrix(other, constant)
        elif Utils.is_scalar(other):
            return self._sum_scalar(other, constant)
        else:
            self._logger.error(
                "'Matrix' instance, int, float or complex expected, not '{}'".format(type(other)))
            raise TypeError(
                "'Matrix' instance, int, float or complex expected, not '{}'".format(type(other)))

    def sum(self, other):
        """Perform a summation with another matrix (element-wise) or scalar (number), i.e., int, float or complex.

        Parameters
        ----------
        other : :py:class:`sparkquantum.math.matrix.Matrix` or scalar
            The other matrix or scalar.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            The resulting matrix.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.math.matrix.Matrix` object or not a scalar.

        """
        rdd, shape, data_type, num_elements = self._sum(other, 1)

        return Matrix(rdd, shape, data_type=data_type,
                      num_elements=num_elements)

    def subtract(self, other):
        """Perform a subtraction with another matrix (element-wise) or scalar (number), i.e., int, float or complex.

        Parameters
        ----------
        other : :py:class:`sparkquantum.math.matrix.Matrix` or scalar
            The other matrix or number.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            The resulting matrix.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.math.matrix.Matrix` object or not a scalar.

        """
        rdd, shape, data_type, num_elements = self._sum(other, -1)

        return Matrix(rdd, shape, data_type=data_type,
                      num_elements=num_elements)

    def _multiply_matrix(self, other):
        if self._shape[1] != other.shape[0]:
            self._logger.error(
                "incompatible shapes {} and {}".format(
                    self._shape, other.shape))
            raise ValueError(
                "incompatible shapes {} and {}".format(
                    self._shape, other.shape))

        new_shape = (self._shape[0], other.shape[1])

        data_type = Utils.get_precedent_type(self._data_type, other.data_type)

        rdd = self._data
        other_rdd = other.data

        if self._num_elements is not None or other.num_elements is not None:
            if self._num_elements is not None and other.num_elements is not None:
                expected_elements = min(self._num_elements, other.num_elements)
            elif self._num_elements is not None:
                expected_elements = self._num_elements
            else:
                expected_elements = other.num_elements

            num_partitions = Utils.get_num_partitions(
                self._spark_context,
                Utils.get_size_of_type(data_type) * expected_elements
            )
        else:
            expected_elements = None

            num_partitions = max(
                rdd.getNumPartitions(),
                other_rdd.getNumPartitions())

        rdd = rdd.join(
            other_rdd, numPartitions=num_partitions
        ).map(
            lambda m: ((m[1][0][0], m[1][1][0]), m[1][0][1] * m[1][1][1])
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        ).map(
            lambda m: (m[0][0], m[0][1], m[1])
        )

        rdd = Utils.remove_zeros(rdd, data_type, Utils.MatrixCoordinateDefault)

        return rdd, new_shape, data_type, expected_elements

    def _multiply_scalar(self, other):
        if (isinstance(other, int) and other == 1 or
                isinstance(other, float) and other == 1.0):
            return self._data, self._shape, self._data_type

        data_type = Utils.get_precedent_type(self._data_type, type(other))

        rdd = self._data

        rdd = rdd.map(
            lambda m: (m[0], m[1], m[2] * other)
        )

        rdd = Utils.remove_zeros(rdd, data_type, Utils.MatrixCoordinateDefault)

        return rdd, self._shape, data_type, self._num_elements

    def multiply(self, other):
        """Multiply this matrix by another one or by a scalar (number), i.e, int, float or complex.

        Notes
        -----
        The coordinate format of this matrix must be :py:const:`sparkquantum.utils.Utils.MatrixCoordinateMultiplier`
        and the other matrix must be :py:const:`sparkquantum.utils.Utils.MatrixCoordinateMultiplicand`.
        The coordinate format of the resulting matrix is :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault`.

        Parameters
        ----------
        other : :py:class:`sparkquantum.math.matrix.Matrix` or a scalar
            The other matrix or scalar that will be multiplied by this matrix.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            The resulting matrix.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.math.matrix.Matrix` or not a scalar.
        ValueError
            If this matrix's and `other`'s shapes are incompatible for multiplication.

        """
        if is_matrix(other):
            rdd, shape, data_type, num_elements = self._multiply_matrix(other)

            return Matrix(rdd, shape, data_type=data_type,
                          num_elements=num_elements)
        elif Utils.is_scalar(other):
            rdd, shape, data_type, num_elements = self._multiply_scalar(other)

            return Matrix(rdd, shape, data_type=data_type,
                          num_elements=num_elements)
        else:
            self._logger.error(
                "'Matrix' intance, int, float or complex expected, not '{}'".format(type(other)))
            raise TypeError(
                "'Matrix' intance, int, float or complex expected, not '{}'".format(type(other)))

    @staticmethod
    def is_empty(matrix):
        """Check whether matrix is empty, i.e., its rows and column sizes are zero.

        Parameters
        ----------
        matrix : :py:class:`sparkquantum.math.matrix.Matrix`
            A :py:class:`sparkquantum.math.matrix.Matrix` object.

        Returns
        -------
        bool
            True if matrix is empty, False otherwise.

        Raises
        ------
        TypeError
            If `matrix` is not a :py:class:`sparkquantum.math.matrix.Matrix` object.

        """
        if not is_matrix(matrix):
            raise TypeError(
                'Matrix instance expected, not {}'.format(type(matrix)))

        return matrix.shape[0] == 0 and matrix.shape[1] == 0

    @staticmethod
    def is_square(matrix):
        """Check whether matrix is square, i.e., its rows and column sizes are equal.

        Parameters
        ----------
        matrix : :py:class:`sparkquantum.math.matrix.Matrix`
            A :py:class:`sparkquantum.math.matrix.Matrix` object.

        Returns
        -------
        bool
            True if matrix is square, False otherwise.

        Raises
        ------
        TypeError
            If `matrix` is not a :py:class:`sparkquantum.math.matrix.Matrix` object.

        """
        if not is_matrix(matrix):
            raise TypeError(
                'Matrix instance expected, not {}'.format(type(matrix)))

        return matrix.shape[0] == matrix.shape[1]

    @staticmethod
    def is_rowvector(matrix):
        """Check whether matrix is a row vector, i.e., it has only one row.

        Parameters
        ----------
        matrix : :py:class:`sparkquantum.math.matrix.Matrix`
            A :py:class:`sparkquantum.math.matrix.Matrix` object.

        Returns
        -------
        bool
            True if matrix is a row vector, False otherwise.

        Raises
        ------
        TypeError
            If `matrix` is not a :py:class:`sparkquantum.math.matrix.Matrix` object.

        """
        if not is_matrix(matrix):
            raise TypeError(
                'Matrix instance expected, not {}'.format(type(matrix)))

        return matrix.shape[0] == 1

    @staticmethod
    def is_columnvector(matrix):
        """Check whether matrix is a column vector, i.e., it has only one column.

        Parameters
        ----------
        matrix : :py:class:`sparkquantum.math.matrix.Matrix`
            A :py:class:`sparkquantum.math.matrix.Matrix` object.

        Returns
        -------
        bool
            True if matrix is a column vector, False otherwise.

        Raises
        ------
        TypeError
            If `matrix` is not a :py:class:`sparkquantum.math.matrix.Matrix` object.

        """
        if not is_matrix(matrix):
            raise TypeError(
                'Matrix instance expected, not {}'.format(type(matrix)))

        return matrix.shape[1] == 1


def is_matrix(obj):
    """Check whether argument is a :py:class:`sparkquantum.math.matrix.Matrix` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.math.matrix.Matrix` object, False otherwise.

    """
    return isinstance(obj, Matrix)
