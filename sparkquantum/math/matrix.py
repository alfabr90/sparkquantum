import math

import numpy as np
from pyspark import RDD, SparkContext

from sparkquantum import conf, constants, util
from sparkquantum.base import Base
from sparkquantum.math import util as mathutil

__all__ = ['Matrix', 'is_matrix']


class Matrix(Base):
    """Class for matrices."""

    def __init__(self, rdd, shape,
                 dtype=complex, coord_format=constants.MatrixCoordinateDefault, nelem=None):
        """Build a matrix object.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this object. Must be 2-dimensional.
        dtype : type, optional
            The Python type of all values in this object. Default value is complex.
        coord_format : int, optional
            The coordinate format of this object. Default value is :py:const:`sparkquantum.constants.MatrixCoordinateDefault`.
        nelem : int, optional
            The expected (or definitive) number of elements. This helps to find a
            better number of partitions when (re)partitioning the RDD. Default value is None.

        """
        if not mathutil.is_shape(shape, ndim=2):
            raise ValueError("invalid shape")

        super().__init__(rdd, nelem=nelem)

        self._shape = shape
        self._dtype = dtype

        self._size = self._shape[0] * self._shape[1]

        self._coord_format = coord_format

    @property
    def shape(self):
        """tuple"""
        return self._shape

    @property
    def dtype(self):
        """type"""
        return self._dtype

    @property
    def size(self):
        """int"""
        return self._size

    @property
    def coord_format(self):
        """int"""
        return self._coord_format

    def __str__(self):
        return '{} with shape {}'.format(self.__class__.__name__, self._shape)

    def _sum_matrix(self, other, constant):
        if self._shape[0] != other.shape[0] or self._shape[1] != other.shape[1]:
            self._logger.error(
                "incompatible shapes {} and {}".format(self._shape, other.shape))
            raise ValueError(
                "incompatible shapes {} and {}".format(self._shape, other.shape))

        dtype = util.get_precedent_type(
            self._dtype, other.dtype)

        rdd = self._data
        other_rdd = other.data

        if self._nelem is not None and other.nelem is not None:
            nelem = self._nelem + other.nelem

            num_partitions = util.get_num_partitions(
                self._sc,
                util.get_size_of_type(dtype) * nelem
            )
        else:
            nelem = None

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

        return Matrix(rdd, self._shape, dtype=dtype, nelem=nelem).clear()

    def _sum_scalar(self, other, constant):
        if other == type(other)():
            return self._data, self._shape, self._dtype, self._nelem

        dtype = util.get_precedent_type(
            self._dtype, type(other))

        rdd = self._data

        other_rdd = self._sc.range(
            self._shape[0]
        ).cartesian(
            self._sc.range(self._shape[1])
        ).map(
            lambda m: (m, constant * other)
        )

        nelem = self._size

        num_partitions = util.get_num_partitions(
            self._sc,
            util.get_size_of_type(dtype) * nelem
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

        return Matrix(rdd, self._shape, dtype=dtype, nelem=nelem).clear()

    def _multiply_matrix(self, other):
        if (self._coord_format != constants.MatrixCoordinateMultiplier or
                other.coord_format != constants.MatrixCoordinateMultiplicand):
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        if self._shape[1] != other.shape[0]:
            self._logger.error(
                "incompatible shapes {} and {}".format(self._shape, other.shape))
            raise ValueError(
                "incompatible shapes {} and {}".format(self._shape, other.shape))

        shape = (self._shape[0], other.shape[1])

        dtype = util.get_precedent_type(self._dtype, other.dtype)

        rdd = self._data
        other_rdd = other.data

        if self._nelem is not None or other.nelem is not None:
            if self._nelem is not None and other.nelem is not None:
                nelem = min(self._nelem, other.nelem)
            elif self._nelem is not None:
                nelem = self._nelem
            else:
                nelem = other.nelem
        else:
            nelem = None

        rdd = rdd.join(
            other_rdd
        ).map(
            lambda m: ((m[1][0][0], m[1][1][0]), m[1][0][1] * m[1][1][1])
        ).reduceByKey(
            lambda a, b: a + b
        ).map(
            lambda m: (m[0][0], m[0][1], m[1])
        )

        return Matrix(rdd, shape, dtype=dtype, nelem=nelem)

    def _multiply_scalar(self, other, constant):
        if self._coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        dtype = util.get_precedent_type(self._dtype, type(other))

        if other == type(other)():
            if constant < 0:
                # It is a division by zero operation
                raise ZeroDivisionError
            elif constant > 0:
                return Matrix(self._sc.emptyRDD, self._shape,
                              dtype=dtype, coord_format=self._coord_format, nelem=0)
            else:
                return Matrix(self._data, self._shape,
                              dtype=dtype, coord_format=self._coord_format, nelem=self._nelem)
        elif other == type(other)() + 1:
            return Matrix(self._data, self._shape,
                          dtype=dtype, coord_format=self._coord_format, nelem=self._nelem)

        rdd = self._data.map(
            lambda m: (m[0], m[1], m[2] * other ** constant)
        )

        return Matrix(rdd, self._shape,
                      dtype=dtype, coord_format=self._coord_format, nelem=self._nelem)

    def sparsity(self):
        """Calculate the sparsity of this matrix.

        Returns
        -------
        float
            The sparsity of this matrix.

        """
        nelem = self._nelem

        if nelem is None:
            self._logger.warning(
                "this matrix will be considered as dense as it has not had its number of elements defined")
            nelem = self._size

        return 1.0 - nelem / self._size

    def dump(self, path, glue=None, codec=None, filename=None):
        """Dump this object's RDD to disk in a unique file or in many part-* files.

        Notes
        -----
        Depending on the chosen dumping mode, this method calls the RDD's :py:func:`pyspark.RDD.collect` method.
        This is not suitable for large working sets, as all data may not fit into driver's main memory.
        This method exports the data in the :py:const:`sparkquantum.constants.MatrixCoordinateDefault` format.

        Parameters
        ----------
        path : str
            The path where the dumped RDD will be located at.
        glue : str, optional
            The glue string that connects each coordinate and value of each element in the RDD.
            Default value is None. In this case, it uses the 'sparkquantum.dumpingGlue' configuration value.
        codec : str, optional
            Codec name used to compress the dumped data.
            Default value is None. In this case, it uses the 'sparkquantum.dumpingCompressionCodec' configuration value.
        filename : str, optional
            File name used when the dumping mode is in a single file. Default value is None.
            In this case, a temporary named file is generated inside the informed path.

        Raises
        ------
        NotImplementedError
            If the coordinate format is not :py:const:`sparkquantum.constants.MatrixCoordinateDefault`.

        ValueError
            If the chosen 'sparkquantum.math.dumpingMode' configuration is not valid.

        """
        if self._coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        if glue is None:
            glue = conf.get(
                self._sc,
                'sparkquantum.dumpingGlue')

        if codec is None:
            codec = conf.get(
                self._sc,
                'sparkquantum.dumpingCompressionCodec')

        dumping_mode = int(
            conf.get(
                self._sc,
                'sparkquantum.math.dumpingMode'))

        rdd = self.clear().data

        rdd = rdd.map(
            lambda m: glue.join((str(m[0]), str(m[1]), str(m[2])))
        )

        if dumping_mode == constants.DumpingModeUniqueFile:
            data = rdd.collect()

            util.create_dir(path)

            if not filename:
                filename = util.get_temp_path(path)
            else:
                filename = util.append_slash(path) + filename

            if len(data):
                with open(filename, 'a') as f:
                    for d in data:
                        f.write(d + "\n")
        elif dumping_mode == constants.DumpingModePartFiles:
            rdd.saveAsTextFile(path, codec)
        else:
            self._logger.error("invalid dumping mode")
            raise ValueError("invalid dumping mode")

    def ndarray(self):
        """Create a Numpy array containing this object's RDD data.

        Notes
        -----
        This method calls the :py:func:`pyspark.RDD.collect` method. This is not suitable for large working sets,
        as all data may not fit into main memory.

        Returns
        -------
        :py:class:`numpy.ndarray`
            The Numpy array.

        Raises
        ------
        NotImplementedError
            If this object's coordinate format is not :py:const:`sparkquantum.constants.MatrixCoordinateDefault`..

        """
        if self._coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        data = self.clear().data.collect()

        result = np.zeros(self._shape, dtype=self._dtype)

        for e in data:
            result[e[0], e[1]] = e[2]

        return result

    def clear(self):
        """Remove possible zero entries of this matrix object.

        Notes
        -----
        Due to the immutability of RDD, a new RDD instance is created.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            A new matrix object.

        Raises
        ------
        NotImplementedError
            If this object's coordinate format is not :py:const:`sparkquantum.constants.MatrixCoordinateDefault`..

        """
        if self._coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        zero = self._dtype()

        rdd = self._data.filter(
            lambda m: m[2] is not None and m[2] != zero
        )

        return Matrix(rdd, self._shape,
                      dtype=self._dtype, coord_format=self._coord_format, nelem=self._nelem)

    def copy(self):
        """Make a copy of this object.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            A new matrix object.

        """
        rdd = self._data.map(
            lambda m: m
        )

        return Matrix(rdd, self._shape,
                      dtype=self._dtype, coord_format=self._coord_format, nelem=self._nelem)

    def to_coordinate(self, coord_format):
        """Change the coordinate format of this object.

        Notes
        -----
        Due to the immutability of RDD, a new RDD instance is created
        in the desired coordinate format.

        Parameters
        ----------
        coord_format : int
            The new coordinate format for this object.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            A new matrix object with the RDD in the desired coordinate format.

        """
        if self._coord_format == coord_format:
            return self

        rdd = self._data

        if self._coord_format != constants.MatrixCoordinateDefault:
            if self._coord_format == constants.MatrixCoordinateMultiplier:
                rdd = rdd.map(
                    lambda m: (m[1][0], m[0], m[1][1])
                )
            elif self._coord_format == constants.MatrixCoordinateMultiplicand:
                rdd = rdd.map(
                    lambda m: (m[0], m[1][0], m[1][1])
                )
            elif self._coord_format == constants.MatrixCoordinateIndexed:
                rdd = rdd.map(
                    lambda m: (m[0][0], m[0][1], m[1])
                )
            else:
                raise ValueError("invalid coordinate format")

        if coord_format != constants.MatrixCoordinateDefault:
            if coord_format == constants.MatrixCoordinateMultiplier:
                rdd = rdd.map(
                    lambda m: (m[1], (m[0], m[2]))
                )
            elif coord_format == constants.MatrixCoordinateMultiplicand:
                rdd = rdd.map(
                    lambda m: (m[0], (m[1], m[2]))
                )
            elif coord_format == constants.MatrixCoordinateIndexed:
                rdd = rdd.map(
                    lambda m: ((m[0], m[1]), m[2])
                )
            else:
                raise ValueError("invalid coordinate format")

        return Matrix(rdd, self._shape,
                      dtype=self._dtype, coord_format=coord_format, nelem=self._nelem)

    def transpose(self):
        """Transpose this matrix.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            The resulting matrix.

        """
        if self._coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        shape = (self._shape[1], self._shape[0])

        rdd = rdd.map(
            lambda m: (m[1], m[0], m[2])
        )

        return Matrix(rdd, shape,
                      dtype=self._dtype, coord_format=self._coord_format, nelem=self._nelem)

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
                "'Matrix' instance expected, not '{}'".format(type(other)))
            raise TypeError(
                "'Matrix' instance expected, not '{}'".format(type(other)))

        if (self._coord_format != constants.MatrixCoordinateDefault or
                other.coord_format != constants.MatrixCoordinateDefault):
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        other_shape = other.shape
        shape = (self._shape[0] * other_shape[0],
                 self._shape[1] * other_shape[1])

        dtype = util.get_precedent_type(self._dtype, other.dtype)

        rdd = self._data
        other_rdd = other.data

        if self._nelem is not None and other.nelem is not None:
            nelem = self._nelem * other.nelem
        else:
            nelem = None

        if self._nelem is not None:
            num_partitions = util.get_num_partitions(
                self._sc,
                util.get_size_of_type(dtype) * self._nelem
            )
        else:
            num_partitions = rdd.getNumPartitions()

        if other.nelem is not None:
            other_num_partitions = util.get_num_partitions(
                self._sc,
                util.get_size_of_type(dtype) * other.nelem
            )
        else:
            other_num_partitions = other_rdd.getNumPartitions()

        num_partitions = num_partitions * other_num_partitions

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

        return Matrix(rdd, shape,
                      dtype=dtype, coord_format=self._coord_format, nelem=nelem)

    def trace(self):
        """Calculate the trace of this matrix.

        Returns
        -------
        float
            The trace of this matrix.

        Raises
        ------
        TypeError
            If this matrix is not square.

        """
        if not Matrix.is_square(self):
            self._logger.error(
                "cannot calculate the trace of non square matrix")
            raise TypeError("cannot calculate the trace of non square matrix")

        if self._coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        return self._data.filter(
            lambda m: m[0] == m[1]
        ).reduce(
            lambda a, b: a[2] + b[2]
        )

    def norm(self):
        """Calculate the norm of this matrix.

        Returns
        -------
        float
            The norm of this matrix.

        """
        if self._coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        if self._dtype == complex:
            def __map(m):
                return m[2].real ** 2 + m[2].imag ** 2
        else:
            def __map(m):
                return m[2] ** 2

        n = self._data.map(
            __map
        ).reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

    def is_unitary(self):
        """Check if this matrix is unitary by calculating its norm.

        Notes
        -----
        This method uses the 'sparkquantum.math.roundPrecision' configuration to round the calculated norm.

        Returns
        -------
        bool
            True if the norm of this matrix is 1.0, False otherwise.

        """
        if self._coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        round_precision = int(
            conf.get(
                self._sc,
                'sparkquantum.math.roundPrecision'))

        return round(self.norm(), round_precision) == 1.0

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
        if is_matrix(other):
            return self._sum_matrix(other, 1)
        elif mathutil.is_scalar(other):
            return self._sum_scalar(other, 1)
        else:
            self._logger.error(
                "'Matrix' instance, int, float or complex expected, not '{}'".format(type(other)))
            raise TypeError(
                "'Matrix' instance, int, float or complex expected, not '{}'".format(type(other)))

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
        if is_matrix(other):
            return self._sum_matrix(other, -1)
        elif mathutil.is_scalar(other):
            return self._sum_scalar(other, -1)
        else:
            self._logger.error(
                "'Matrix' instance, int, float or complex expected, not '{}'".format(type(other)))
            raise TypeError(
                "'Matrix' instance, int, float or complex expected, not '{}'".format(type(other)))

    def multiply(self, other):
        """Multiply this matrix by another one or by a scalar (number), i.e, int, float or complex.

        Notes
        -----
        The coordinate format of this matrix must be :py:const:`sparkquantum.constants.MatrixCoordinateMultiplier`
        and the other matrix must be :py:const:`sparkquantum.constants.MatrixCoordinateMultiplicand`.
        The coordinate format of the resulting matrix is :py:const:`sparkquantum.constants.MatrixCoordinateDefault`.

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
            return self._multiply_matrix(other)
        elif mathutil.is_scalar(other):
            return self._multiply_scalar(other, 1)
        else:
            self._logger.error(
                "'Matrix' intance, int, float or complex expected, not '{}'".format(type(other)))
            raise TypeError(
                "'Matrix' intance, int, float or complex expected, not '{}'".format(type(other)))

    def divide(self, other):
        """Divide this matrix by another one or by a scalar (number), i.e, int, float or complex.

        Parameters
        ----------
        other : :py:class:`sparkquantum.math.matrix.Matrix` or a scalar
            The other matrix or scalar that will divide this matrix.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            The resulting matrix.

        Raises
        ------
        NotImplementedError
            If `other` is a :py:class:`sparkquantum.math.matrix.Matrix`.
        TypeError
            If `other` is not a scalar.
        ZeroDivisionError
            if `other` is equal to zero.

        """
        if is_matrix(other):
            raise NotImplementedError
        elif mathutil.is_scalar(other):
            return self._multiply_scalar(other, -1)
        else:
            self._logger.error(
                "'Matrix' intance, int, float or complex expected, not '{}'".format(type(other)))
            raise TypeError(
                "'Matrix' intance, int, float or complex expected, not '{}'".format(type(other)))

    def dot_product(self, other):
        """Perform a dot (scalar) product with another matrix (column vector).

        Notes
        -----
        This matrix must be a row vector and `other` must be a column vector.

        Parameters
        ----------
        other : :py:class:`sparkquantum.math.matrix.Matrix`
            The other matrix.

        Returns
        -------
        int, float or complex

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.math.matrix.Matrix`.

        """
        if not self.is_rowvector(self):
            self._logger.error("this 'Matrix' instance must be a row vector")
            raise ValueError("this 'Matrix' instance must be a row vector")

        if not self.is_columnvector(self):
            self._logger.error(
                "the other 'Matrix' instance must be a column vector")
            raise ValueError(
                "the other 'Matrix' instance must be a column vector")

        matrix = self.multiply(other)

        result = matrix.data.collect()

        if len(result) == 0:
            return matrix.dtype()
        else:
            return result[0]

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

        """
        return (is_matrix(matrix)
                and matrix.shape[0] == 0 and matrix.shape[1] == 0)

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

        """
        return is_matrix(matrix) and matrix.shape[0] == matrix.shape[1]

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

        """
        return is_matrix(matrix) and matrix.shape[0] == 1

    @staticmethod
    def is_colvector(matrix):
        """Check whether matrix is a column vector, i.e., it has only one column.

        Parameters
        ----------
        matrix : :py:class:`sparkquantum.math.matrix.Matrix`
            A :py:class:`sparkquantum.math.matrix.Matrix` object.

        Returns
        -------
        bool
            True if matrix is a column vector, False otherwise.

        """
        return is_matrix(matrix) and matrix.shape[1] == 1

    @staticmethod
    def diagonal(size, value):
        """Create a diagonal matrix with its elements being the desired value.

        Parameters
        ----------
        size : int
            The size of the diagonal.
        value: int, float or complex
            The value of each element of the diagonal matrix.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            The resulting matrix.

        Raises
        ------
        TypeError
            If `size` is not an int or `value` is not a scalar (number).

        """
        if not isinstance(size, int):
            raise TypeError("int expected, not {}".format(type(size)))

        if not mathutil.is_scalar(value):
            raise TypeError(
                "int, float or complex expected, not {}".format(type(value)))

        sc = SparkContext.getOrCreate()

        shape = (size, size)
        dtype = type(value)

        nelem = shape[0]

        if value == dtype():
            rdd = sc.emptyRDD()
        else:
            num_partitions = util.get_num_partitions(
                sc,
                util.get_size_of_type(dtype) * nelem
            )

            rdd = sc.range(size, numSlices=num_partitions).map(
                lambda m: (m, m, value)
            )

        return Matrix(rdd, shape, dtype=dtype, nelem=nelem)

    @staticmethod
    def eye(size):
        """Create an identity matrix.

        Parameters
        ----------
        size : int
            The size of the diagonal.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            The resulting matrix.

        Raises
        ------
        TypeError
            If `size` is not an int.

        """
        return Matrix.diagonal(size, 1.0)

    @staticmethod
    def zeros(shape, dtype=float):
        """Create a matrix full of zeros.

        Notes
        -----
        As all matrix-like objects are treated as sparse, an empty RDD is used.

        Parameters
        ----------
        shape : tuple
            The shape of the matrix.
        dtype : type, optional
            The Python type of all values in this object. Default value is float.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            The resulting matrix.

        Raises
        ------
        TypeError
            If `shape` is not a valid shape.

        """
        if not mathutil.is_shape(shape, ndim=2):
            raise ValueError("invalid shape")

        sc = SparkContext.getOrCreate()

        nelem = 0

        rdd = sc.emptyRDD()

        return Matrix(rdd, shape, dtype=dtype, nelem=nelem)

    @staticmethod
    def ones(shape, dtype=float):
        """Create a matrix full of ones.

        Parameters
        ----------
        shape : tuple
            The shape of the matrix.
        dtype : type, optional
            The Python type of all values in this object. Default value is float.

        Returns
        -------
        :py:class:`sparkquantum.math.matrix.Matrix`
            The resulting matrix.

        Raises
        ------
        TypeError
            If `shape` is not a valid shape.

        """
        if not mathutil.is_shape(shape, ndim=2):
            raise ValueError("invalid shape")

        sc = SparkContext.getOrCreate()

        value = dtype() + 1

        nelem = shape[0] * shape[1]

        num_partitions = util.get_num_partitions(
            sc,
            util.get_size_of_type(dtype) * nelem
        )

        rdd = sc.range(
            shape[0], numSlices=num_partitions
        ).cartesian(
            sc.range(shape[1], numSlices=num_partitions)
        ).map(
            lambda m: (m[0], m[1], value)
        )

        return Matrix(rdd, shape, dtype=dtype, nelem=nelem)


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
