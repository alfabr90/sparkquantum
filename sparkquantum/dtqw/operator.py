from sparkquantum.dtqw.state import State, is_state
from sparkquantum.math.matrix import Matrix
from sparkquantum.utils.utils import Utils

__all__ = ['Operator', 'is_operator']


class Operator(Matrix):
    """Class for the operators of quantum walks."""

    def __init__(self, rdd, shape, data_type=complex,
                 coordinate_format=Utils.MatrixCoordinateDefault, num_elements=None):
        """Build an operator object.

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
        super().__init__(
            rdd,
            shape,
            data_type=data_type,
            coordinate_format=coordinate_format,
            num_elements=num_elements)

    def __str__(self):
        return '{} with shape {}'.format(self.__class__.__name__, self._shape)

    def change_coordinate(self, coordinate_format):
        """Change the coordinate format of this object.

        Notes
        -----
        Due to the immutability of RDD, a new RDD instance is created
        in the desired coordinate format. Thus, a new instance of this class
        is returned with this RDD.

        Parameters
        ----------
        coordinate_format : int
            The new coordinate format of this object.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator`
            A new operator object with the RDD in the desired coordinate format.

        """
        rdd = self._change_coordinate(coordinate_format)

        return Operator(rdd, self._shape, data_type=self._data_type,
                        coordinate_format=coordinate_format, num_elements=self._num_elements)

    def transpose(self):
        """Transpose this operator.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator`
            The resulting operator.

        """
        rdd, shape = self._transpose()

        return Operator(rdd, shape, data_type=self._data_type,
                        num_elements=self._num_elements)

    def kron(self, other):
        """Perform a tensor (Kronecker) product with another operator.

        Parameters
        ----------
        other : :py:class:`sparkquantum.dtqw.operator.Operator`
            The other operator.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator`
            The resulting operator.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.dtqw.operator.Operator`.

        """
        if not is_operator(other):
            self._logger.error(
                "'Operator' instance expected, not '{}'".format(type(other)))
            raise TypeError(
                "'Operator' instance expected, not '{}'".format(type(other)))

        rdd, shape, data_type, num_elements = self._kron(other)

        return Operator(rdd, shape, data_type=data_type,
                        num_elements=num_elements)

    def sum(self, other):
        return None

    def subtract(self, other):
        return None

    def multiply(self, other):
        """Multiply this operator with another one or with a system state.

        Parameters
        ----------
        other : :py:class:`sparkquantum.dtqw.operator.Operator` or :py:class:`sparkquantum.dtqw.state.State`
            An operator if multiplying another operator, state otherwise.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator` or :py:class:`sparkquantum.dtqw.state.State`
            :py:class:`sparkquantum.dtqw.operator.Operator` if multiplying another operator, :py:class:`sparkquantum.dtqw.state.State` otherwise.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.dtqw.operator.Operator` nor :py:class:`sparkquantum.dtqw.state.State`.

        ValueError
            If this matrix's and `other`'s shapes are incompatible for multiplication.

        """
        if is_operator(other):
            rdd, shape, data_type, num_elements = self._multiply_matrix(other)

            return Operator(rdd, shape, data_type=data_type,
                            num_elements=num_elements)
        elif is_state(other):
            rdd, shape, data_type, num_elements = self._multiply_matrix(other)

            return State(rdd, shape, other.coin, other.mesh, other.num_particles,
                         interaction=other.interaction, data_type=data_type, num_elements=num_elements)
        else:
            self._logger.error(
                "'State' or 'Operator' instance expected, not '{}'".format(type(other)))
            raise TypeError(
                "'State' or 'Operator' instance expected, not '{}'".format(type(other)))


def is_operator(obj):
    """Check whether argument is an :py:class:`sparkquantum.dtqw.operator.Operator` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is an :py:class:`sparkquantum.dtqw.operator.Operator` object, False otherwise.

    """
    return isinstance(obj, Operator)
