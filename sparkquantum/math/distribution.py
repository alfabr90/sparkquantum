import math

import numpy as np

from sparkquantum.base import Base

__all__ = [
    'ProbabilityDistribution',
    'RandomVariable',
    'is_probability_distribution']


class RandomVariable:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def __str__(self):
        return "Random Variable with name '{}'".format(self._name)


class ProbabilityDistribution(Base):
    """Class for probability distributions."""

    def __init__(self, rdd, shape, variables, num_elements=None):
        """Build a probability distribution object.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this object. Must be a n-dimensional tuple.
        variables : list or tuple of :py:class:`sparkquantum.math.distribution.RandomVariable`
            The random variables of this probability distribution.
        num_elements : int, optional
            The expected (or definitive) number of elements. This helps to find a
            better number of partitions when (re)partitioning the RDD. Default value is None.

        """
        super().__init__(rdd, num_elements=num_elements)

        self._shape = shape
        self._variables = variables
        self._data_type = float

        if not isinstance(shape, (list, tuple)):
            self._logger.error("invalid shape")
            raise ValueError("invalid shape")

        self._size = 1

        for s in shape:
            self._size *= s

        if not isinstance(self._variables, (list, tuple)):
            self._logger.error(
                "list or tuple expected, not {}".format(type(self._variables)))
            raise TypeError(
                "list or tuple expected, not {}".format(type(self._variables)))
        elif len(self._variables) < 1:
            self._logger.error(
                "invalid number of random variables. It must be greater than or equal to 1")
            raise ValueError(
                "invalid number of random variables. It must be greater than or equal to 1")

    @property
    def shape(self):
        """tuple"""
        return self._shape

    @property
    def variables(self):
        """list or tuple of :py:class:`sparkquantum.math.distribution.RandomVariable`"""
        return self._variables

    @property
    def size(self):
        """int"""
        return self._size

    def __str__(self):
        return 'Probability Distribution of random variables {}'.format(
            [v.name for v in self._variables])

    def ndarray(self):
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
        data = self._data.collect()

        result = np.zeros(self._shape, dtype=self._data_type)

        for e in data:
            result[e[0:-1]] = e[-1]

        return result

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
    """Check whether argument is a :py:class:`sparkquantum.math.distribution.ProbabilityDistribution` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.math.distribution.ProbabilityDistribution` object, False otherwise.

    """
    return isinstance(obj, ProbabilityDistribution)
