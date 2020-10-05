from pyspark import SparkContext

from sparkquantum import constants, util
from sparkquantum.dtqw.operator import Operator

__all__ = ['Coin']


class Coin:
    """Top-level class for coins."""

    def __init__(self):
        """Build a top-level coin object.

        """
        self._sc = SparkContext.getOrCreate()

        self._data = None

        self._logger = util.get_logger(
            self._sc, self.__class__.__name__)

    @property
    def sc(self):
        """:py:class:`pyspark.SparkContext`"""
        return self._sc

    @property
    def data(self):
        """tuple"""
        return self._data

    def __del__(self):
        # In cases where multiple simulations are performed,
        # the same Python's logger object is used for the same class name.
        # Removing all its handlers, the object is reset.
        for h in self._logger.handlers:
            self._logger.removeHandler(h)

    def __str__(self):
        """Build a string representing this coin.

        Returns
        -------
        str
            The string representation of this coin.

        """
        return self.__class__.__name__

    def create_operator(self, pspace,
                        repr_format=constants.StateRepresentationFormatCoinPosition):
        """Build a coin operator for a quantum walk, multiplying this coin data by
        an identity matrix representing the position space.

            ``C = C0 (X) I`` or ``C = I (X) C0``

        Parameters
        ----------
        pspace : int
            The size of the position space.
        repr_format : int, optional
            Indicate how the quantum system is represented.
            Default value is :py:const:`sparkquantum.constants.StateRepresentationFormatCoinPosition`.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator`
            The created operator using this coin.

        Raises
        ------
        ValueError
            If `repr_format` is not valid.

        """
        cspace = len(self._data)
        shape = (cspace * pspace, cspace * pspace)

        data = util.broadcast(self._sc, self._data)

        nelem = cspace ** 2 * pspace

        if repr_format == constants.StateRepresentationFormatCoinPosition:
            # The coin operator is built by applying a tensor product between
            # the chosen coin and an identity matrix representing the position
            # space.
            def __map(p):
                for i in range(cspace):
                    for j in range(cspace):
                        yield (i * pspace + p, j * pspace + p, data.value[i][j])
        elif repr_format == constants.StateRepresentationFormatPositionCoin:
            # The coin operator is built by applying a tensor product between
            # an identity matrix representing the position space and the chosen
            # coin.
            def __map(p):
                for i in range(cspace):
                    for j in range(cspace):
                        yield (p * cspace + i, p * cspace + j, data.value[i][j])
        else:
            self._logger.error("invalid representation format")
            raise ValueError("invalid representation format")

        rdd = self._sc.range(
            pspace
        ).flatMap(
            __map
        )

        return Operator(rdd, shape, nelem=nelem)


def is_coin(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.coin.coin.Coin` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.coin.coin.Coin` object, False otherwise.

    """
    return isinstance(obj, Coin)
