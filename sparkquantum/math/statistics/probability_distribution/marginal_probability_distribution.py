from sparkquantum.math.statistics.probability_distribution.probability_distribution import ProbabilityDistribution

__all__ = ['MarginalProbabilityDistribution']


class MarginalProbabilityDistribution(ProbabilityDistribution):
    """Class for marginal probability distributions."""

    def __init__(self, rdd, shape, num_elements=None):
        """Build an object for a marginal probability distribution.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a two-dimensional tuple.
        num_elements : int, optional
            The expected (or definitive) number of elements. This helps to find a
            better number of partitions when (re)partitioning the RDD. Default value is None.

        """
        super().__init__(rdd, shape, 1, num_elements=num_elements)

    def __str__(self):
        return 'Marginal Probability Distribution'
