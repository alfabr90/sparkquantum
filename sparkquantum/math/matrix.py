import math
import numpy as np

from sparkquantum.math.base import Base
from sparkquantum.math.vector import Vector, is_vector
from sparkquantum.utils.utils import Utils

__all__ = ['Matrix', 'is_matrix']


class Matrix(Base):
    """Class for matrices."""

    def __init__(self, rdd, shape, data_type=complex,
                 coord_format=Utils.MatrixCoordinateDefault):
        """Build a matrix object.

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
            Default value is :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault`.

        """
        super().__init__(rdd, shape, data_type=data_type)

        self._coordinate_format = coord_format

    @property
    def coordinate_format(self):
        """int"""
        return self._coordinate_format

    def __str__(self):
        return '{} with shape {}'.format(self.__class__.__name__, self._shape)

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
        new_shape = (
            self._shape[0] *
            other_shape[0],
            self._shape[1] *
            other_shape[1])
        data_type = Utils.get_precendent_type(self._data_type, other.data_type)

        expected_elems = self._num_nonzero_elements * other.num_nonzero_elements
        expected_size = Utils.get_size_of_type(data_type) * expected_elems
        num_partitions = Utils.get_num_partitions(
            self.data.context, expected_size)

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
                lambda m: (
                    m[0][1] * other_shape[1] + m[1][1],
                    (m[0][0] * other_shape[0] + m[1][0],
                     m[0][2] * m[1][2]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        elif self._coordinate_format == Utils.MatrixCoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (
                    m[0][0] * other_shape[0] + m[1][0],
                    (m[0][1] * other_shape[1] + m[1][1],
                     m[0][2] * m[1][2]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        else:  # Utils.MatrixCoordinateDefault
            rdd = rdd.map(
                lambda m: (
                    m[0][0] * other_shape[0] + m[1][0],
                    m[0][1] * other_shape[1] + m[1][1],
                    m[0][2] * m[1][2])
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
            Default value is :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault`.

        Returns
        -------
        :py:class:`sparkquantum.math.Matrix`
            The resulting matrix.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.math.Matrix`.

        """
        if not is_matrix(other):
            self._logger.error(
                "'Matrix' instance expected, not '{}'".format(
                    type(other)))
            raise TypeError(
                "'Matrix' instance expected, not '{}'".format(
                    type(other)))

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

    def _multiply_matrix(self, other, coord_format):
        if self._shape[1] != other.shape[0]:
            self._logger.error(
                "incompatible shapes {} and {}".format(
                    self._shape, other.shape))
            raise ValueError(
                "incompatible shapes {} and {}".format(
                    self._shape, other.shape))

        shape = (self._shape[0], other.shape[1])
        num_partitions = max(
            self.data.getNumPartitions(),
            other.data.getNumPartitions())

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
            self._logger.error(
                "incompatible shapes {} and {}".format(
                    self._shape, other.shape))
            raise ValueError(
                "incompatible shapes {} and {}".format(
                    self._shape, other.shape))

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
        other : :py:class:`sparkquantum.math.Matrix` or :py:class:`sparkquantum.math.Vector`
            A :py:class:`sparkquantum.math.Matrix` if multiplying another matrix, :py:class:`sparkquantum.math.Vector` otherwise.
        coord_format : int, optional
            Indicate if the matrix must be returned in an apropriate format for multiplications.
            Default value is :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault`. Not applicable when multiplying a :py:class:`sparkquantum.math.Vector`.

        Returns
        -------
        :py:class:`sparkquantum.math.Matrix` or :py:class:`sparkquantum.math.Vector`
            A :py:class:`sparkquantum.math.Matrix` if multiplying another matrix, :py:class:`sparkquantum.math.Vector` otherwise.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.math.Matrix` nor :py:class:`sparkquantum.math.Vector`.

        ValueError
            If this matrix's and `other`'s shapes are incompatible for multiplication.

        """
        if is_matrix(other):
            return self._multiply_matrix(other, coord_format)
        elif is_vector(other):
            return self._multiply_vector(other)
        else:
            self._logger.error(
                "'Matrix' or 'Vector' instance expected, not '{}'".format(
                    type(other)))
            raise TypeError(
                "'Matrix' or 'Vector' instance expected, not '{}'".format(
                    type(other)))


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
