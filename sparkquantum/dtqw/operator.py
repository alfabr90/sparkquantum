import math
import numpy as np

from sparkquantum.math.matrix import Matrix
from sparkquantum.dtqw.state import State, is_state
from sparkquantum.utils.utils import Utils

__all__ = ['Operator', 'is_operator']


class Operator(Matrix):
    """Class for the operators of quantum walks."""

    def __init__(self, rdd, shape, data_type=complex, coord_format=Utils.MatrixCoordinateDefault):
        """Build an :py:class:`sparkquantum.dtqw.Operator` object.

        Parameters
        ----------
        rdd : RDD
            The base RDD of this object.
        shape : tuple
            The shape of this operator object. Must be a two-dimensional tuple.
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

    def kron(self, other, coord_format=Utils.MatrixCoordinateDefault):
        """Perform a tensor (Kronecker) product with another operator.

        Parameters
        ----------
        other : :py:class:`sparkquantum.dtqw.Operator`
            The other operator.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is :py:const:`Utils.MatrixCoordinateDefault`.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.Operator`
            The resulting operator.

        """
        if not is_operator(other):
            if self._logger:
                self._logger.error("'Operator' instance expected, not '{}'".format(type(other)))
            raise TypeError("'Operator' instance expected, not '{}'".format(type(other)))

        rdd, new_shape = self._kron(other)

        return Operator(rdd, new_shape, coord_format=coord_format)

    def _multiply_operator(self, other, coord_format):
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

        return Operator(rdd, shape, coord_format=coord_format)

    def _multiply_state(self, other):
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

        return State(rdd, shape, other.mesh, other.num_particles)

    def multiply(self, other, coord_format=Utils.MatrixCoordinateDefault):
        """Multiply this operator with another one or with a system state.

        Parameters
        ----------
        other :py:class:`sparkquantum.dtqw.Operator` or `State`
            An operator if multiplying another operator, State otherwise.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is :py:const:`Utils.MatrixCoordinateDefault`. Not applicable when multiplying a State.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.Operator` or `State`
            :py:class:`sparkquantum.dtqw.Operator` if multiplying another operator, `State` otherwise.

        Raises
        ------
        TypeError

        """
        if is_operator(other):
            return self._multiply_operator(other, coord_format)
        elif is_state(other):
            return self._multiply_state(other)
        else:
            if self._logger:
                self._logger.error("'State' or 'Operator' instance expected, not '{}'".format(type(other)))
            raise TypeError("'State' or 'Operator' instance expected, not '{}'".format(type(other)))


def is_operator(obj):
    """Check whether argument is an Operator object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is an Operator object, False otherwise.

    """
    return isinstance(obj, Operator)
