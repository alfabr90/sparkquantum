import math
from datetime import datetime

import numpy as np

from sparkquantum import plot
from sparkquantum.base import Base
from sparkquantum.math.util import is_shape

__all__ = ['ProbabilityDistribution', 'is_distribution']


class ProbabilityDistribution(Base):
    """Class for probability distributions."""

    def __init__(self, rdd, shape, domain, nelem=None):
        """Build a probability distribution object.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this object. Must be 2-dimensional.
        domain : array-like
            The domain where the values of this probability distribution will settle.
        nelem : int, optional
            The expected (or definitive) number of elements. This helps to find a
            better number of partitions when (re)partitioning the RDD. Default value is None.

        """
        if not is_shape(shape, ndim=2):
            raise ValueError("invalid shape")

        super().__init__(rdd, nelem=nelem)

        self._shape = shape
        self._domain = domain

        self._size = self._shape[0] * self._shape[1]

        self._dtype = float

    @property
    def shape(self):
        """tuple"""
        return self._shape

    @property
    def domain(self):
        """tuple of int, float or complex"""
        return tuple(self._domain)

    @property
    def size(self):
        """int"""
        return self._size

    @property
    def dtype(self):
        """type"""
        return self._dtype

    def __str__(self):
        return 'Probability distribution of random variables {}'.format(
            self._variables)

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

        """
        ndim = len(self._domain)

        if ndim == 1 and self._shape[1] == 2:
            # One-dimensional grids with just one random variable
            shape = (max(self._domain[0]) - min(self._domain[0]) + 1, 1)
        elif ndim == 1 and self._shape[1] == 3:
            # One-dimensional grids with two random variables
            shape = (max(self._domain[0]) - min(self._domain[0]) + 1,
                     max(self._domain[0]) - min(self._domain[0]) + 1)
        elif ndim == 2 and self._shape[1] == 3:
            # Two-dimensional grids with one random variable
            shape = (max(self._domain[0]) - min(self._domain[0]) + 1,
                     max(self._domain[1]) - min(self._domain[1]) + 1)
        else:
            self._logger.error(
                "incompatible domain dimension and number of variables to create a Numpy array")
            raise NotImplementedError(
                "incompatible domain dimension and number of variables to create a Numpy array")

        data = self._data.collect()

        result = np.zeros(shape, dtype=self._dtype)

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
        dtype = self._dtype()

        return self.data.filter(
            lambda m: m[-1] != dtype
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
        dtype = self._dtype()

        n = self.data.filter(
            lambda m: m[-1] != dtype
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

    def plot(self, filename, title=None, labels=None, **kwargs):
        """Plot this distribution.

        Parameters
        ----------
        axis: array-like
            The domain values of the plot.
        filename: str
            The filename to save the plot. Must not have an extension.
        title: str, optional
            The title of the plot.
        labels: tuple, optional
            The labels of each axis. Default value is None.
        \*\*kwargs
            Keyword arguments being passed to `matplotlib <https://matplotlib.org>`_.

        """
        ndim = len(self._domain)

        if ndim == 1 and self._shape[1] == 2:
            # One-dimensional grids with just one random variable
            axis = self._domain[0]
        elif ndim == 1 and self._shape[1] == 3:
            # One-dimensional grids with just two random variables
            axis = (self._domain[0], self._domain[0])
        elif ndim == 2 and self._shape[1] == 3:
            # Two-dimensional grids with just one random variable
            axis = self._domain
        else:
            self._logger.error(
                "incompatible domain dimension and number of variables to plot distribution")
            raise NotImplementedError(
                "incompatible domain dimension and number of variables to plot distribution")

        if labels is None:
            labels = [v for v in self._variables]
            labels.append('Probability')

        self._logger.info("starting plot of probabilities...")

        time = datetime.now()

        if self._shape[1] == 2:
            plot.line(axis, self.ndarray(), filename,
                      title=title, labels=labels, **kwargs)
        else:
            plot.surface(axis, self.ndarray(), filename,
                         title=title, labels=labels, **kwargs)

            plot.contour(axis, self.ndarray(), filename + '_contour',
                         title=title, labels=labels, **kwargs)

        time = (datetime.now() - time).total_seconds()

        self._logger.info("plot was done in {}s".format(time))


def is_distribution(obj):
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
