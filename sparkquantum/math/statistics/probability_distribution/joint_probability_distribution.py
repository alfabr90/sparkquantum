from sparkquantum.math.statistics.probability_distribution.probability_distribution import ProbabilityDistribution

__all__ = ['JointProbabilityDistribution']


class JointProbabilityDistribution(ProbabilityDistribution):
    """Class for joint probability distributions."""

    def __init__(self, rdd, shape, num_variables, num_elements=None):
        """Build an object for a joint probability distribution.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a two-dimensional tuple.
        num_variables : int
            The number of variables of this probability distribution.
        num_elements : int, optional
            The expected (or definitive) number of elements. This helps to find a
            better number of partitions when (re)partitioning the RDD. Default value is None.

        """
        super().__init__(rdd, shape, num_variables, num_elements=num_elements)

    def __str__(self):
        if self._num_variables == 1:
            variables = 'one random variable'
        else:
            variables = '{} random variables'.format(self._num_variables)

        return 'Joint Probability Distribution of {}'.format(variables)
