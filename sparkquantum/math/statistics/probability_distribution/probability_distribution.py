import math

from sparkquantum.base import Base

__all__ = ['ProbabilityDistribution', 'is_probability_distribution']


class ProbabilityDistribution(Base):
    """Top-level class for probability distributions."""

    def __init__(self, rdd, shape, num_variables, num_elements=None):
        """Build a top-level object for a probability distribution.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this object. Must be a n-dimensional tuple.
        num_variables : int
            The number of variables of this probability distribution.
        num_elements : int, optional
            The expected (or definitive) number of elements. This helps to find a
            better number of partitions when (re)partitioning the RDD. Default value is None.

        """
        super().__init__(rdd, num_elements=num_elements)

        self._shape = shape
        self._num_variables = num_variables
        self._data_type = float

        self._size = self._shape[0] * self._shape[1]

        if self._num_variables < 1:
            self._logger.error(
                "invalid number of variables. It must be greater than or equal to 1")
            raise ValueError(
                "invalid number of variables. It must be greater than or equal to 1")

    @property
    def shape(self):
        """tuple"""
        return self._shape

    @property
    def num_variables(self):
        """int"""
        return self._num_variables

    @property
    def data_type(self):
        """type"""
        return self._data_type

    @property
    def size(self):
        """int"""
        return self._size

    def __str__(self):
        if self._num_variables == 1:
            variables = 'one random variable'
        else:
            variables = '{} random variables'.format(self._num_variables)

        return 'Probability Distribution of {}'.format(variables)

    def sum(self):
        """Sum the probabilities of this probability distribution.

        Returns
        -------
        float
            The sum of the probabilities.

        """
        data_type = self._data_type()

        return self.data.filter(
            lambda m: m[-1] != data_type
        ).map(
            lambda m: m[-1]
        ).reduce(
            lambda a, b: a + b
        )

    def norm(self):
        """Calculate the norm of this probability distribution.

        Returns
        -------
        float
            The norm of this probability distribution.

        """
        data_type = self._data_type()

        n = self.data.filter(
            lambda m: m[-1] != data_type
        ).map(
            lambda m: m[-1] ** 2
        ).reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

    def max(self):
        """Find the maximum value of this probability distribution.

        Returns
        ------
        float
            The maximum value of this probability distribution.

        """
        return self.data.map(
            lambda m: m[-1]
        ).max()

    def min(self):
        """Find the minimum value of this probability distribution.

        Returns
        ------
        float
            The minimum value of this probability distribution.

        """
        return self.data.map(
            lambda m: m[-1]
        ).min()


def is_probability_distribution(obj):
    """Check whether argument is a :py:class:`sparkquantum.math.statistics.probability_distribution.probability_distribution.ProbabilityDistribution` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.math.statistics.probability_distribution.probability_distribution.ProbabilityDistribution` object, False otherwise.

    """
    return isinstance(obj, ProbabilityDistribution)
