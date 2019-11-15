import math
from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.math.base import Base
from sparkquantum.utils.utils import Utils

__all__ = ['Vector', 'is_vector']


class Vector(Base):
    """Class for vectors."""

    def __init__(self, rdd, shape, data_type=complex):
        """Build a vector object.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this vector object. Must be a two-dimensional tuple.
        data_type : type, optional
            The Python type of all values in this object. Default value is complex.

        """
        super().__init__(rdd, shape, data_type=data_type)

    def _kron(self, other):
        other_shape = other.shape
        new_shape = (self._shape[0] * other_shape[0], 1)

        expected_elems = new_shape[0]
        expected_size = Utils.get_size_of_type(self.data_type) * expected_elems
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

        rdd = rdd.map(
            lambda m: (m[0][0] * other_shape[0] + m[1][0], m[0][1] * m[1][1])
        )

        return rdd, new_shape

    def kron(self, other):
        """Perform a tensor (Kronecker) product with another vector.

        Parameters
        ----------
        other : :py:class:`sparkquantum.math.Vector`
            The other vector.

        Returns
        -------
        :py:class:`sparkquantum.math.Vector`
            The resulting vector.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.math.Vector`.

        """
        if not is_vector(other):
            if self._logger:
                self._logger.error(
                    "'Vector' instance expected, not '{}'".format(
                        type(other)))
            raise TypeError(
                "'Vector' instance expected, not '{}'".format(
                    type(other)))

        rdd, new_shape = self._kron(other)

        return Vector(rdd, new_shape)

    def norm(self):
        """Calculate the norm of this vector.

        Returns
        -------
        float
            The norm of this vector.

        """
        data_type = self._data_type()

        n = self.data.filter(
            lambda m: m[1] != data_type
        ).map(
            lambda m: m[1].real ** 2 + m[1].imag ** 2
        ).reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

    def is_unitary(self):
        """Check if this vector is unitary by calculating its norm.

        Notes
        -----
        This method uses the 'quantum.math.roundPrecision' configuration to round the calculated norm.

        Returns
        -------
        bool
            True if the norm of this vector is 1.0, False otherwise.

        """
        round_precision = int(
            Utils.get_conf(
                self._spark_context,
                'quantum.math.roundPrecision'))

        return round(self.norm(), round_precision) == 1.0


def is_vector(obj):
    """Check whether argument is a :py:class:`sparkquantum.math.Vector` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.math.Vector` object, False otherwise.

    """
    return isinstance(obj, Vector)
