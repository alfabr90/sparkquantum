import math
import numpy as np

from sparkquantum.math.matrix import Matrix, is_matrix
from sparkquantum.dtqw.math.state import State, is_state
from sparkquantum.utils.utils import Utils

__all__ = ['Operator', 'is_operator']


class Operator(Matrix):
    """Class for the operators of quantum walks."""

    def __init__(self, df, shape, data_type=complex):
        """Build an `Operator` object.

        Parameters
        ----------
        df : `DataFrame`
            The base DataFrame of this object.
        shape : tuple
            The shape of this operator object. Must be a 2-dimensional tuple.
        data_type : type, optional
            The Python type of all values in this object. Default value is complex.

        """
        super().__init__(df, shape, data_type=data_type)

    @staticmethod
    def from_matrix(other):
        """Instantiate an operator from an existing matrix object.

        Parameters
        ----------
        other : `Matrix`
            The matrix to be the origin for the new operator.

        Returns
        -------
        `Operator`
            The resulting operator.

        """
        if not is_matrix(other):
            #if self._logger:
            #   self._logger.error("'Matrix' instance expected, not '{}'".format(type(other)))
            raise TypeError("'Matrix' instance expected, not '{}'".format(type(other)))

        return Operator(other.data, other.shape, data_type=other.data_type)

    def kron(self, other):
        """Perform a tensor (Kronecker) product with another operator.

        Parameters
        ----------
        other : `Operator`
            The other operator.

        Returns
        -------
        `Operator`
            The resulting operator.

        """
        if not is_operator(other):
            if self._logger:
                self._logger.error("'Operator' instance expected, not '{}'".format(type(other)))
            raise TypeError("'Operator' instance expected, not '{}'".format(type(other)))

        return Operator.from_matrix(super.kron(other))

    def multiply(self, other):
        """Multiply this operator with another one or with a system state.

        Parameters
        ----------
        other `Operator` or `State`
            An operator if multiplying another operator, State otherwise.

        Returns
        -------
        `Operator` or `State`
            `Operator` if multiplying another operator, `State` otherwise.

        Raises
        ------
        `TypeError`

        """
        if is_operator(other):
            return Operator.from_matrix(super.multiply(other))
        elif is_state(other):
            return State.from_vector(super.multiply(other), other.mesh, other.num_particles)
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
