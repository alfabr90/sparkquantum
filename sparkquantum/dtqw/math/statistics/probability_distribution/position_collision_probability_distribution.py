from sparkquantum.dtqw.math.statistics.probability_distribution.position_probability_distribution import PositionProbabilityDistribution
from sparkquantum.dtqw.math.statistics.probability_distribution.position_joint_probability_distribution import PositionJointProbabilityDistribution

__all__ = ['PositionCollisionProbabilityDistribution']


class PositionCollisionProbabilityDistribution(
        PositionProbabilityDistribution):
    """Class for collision probability distributions regarding the possible positions
    of all particles of a quantum system state when the particles are at the same sites."""

    def __init__(self, rdd, shape, num_variables, state, num_elements=None):
        """Build an object for a collision probability distribution regarding the possible
        positions of all particles of a quantum system state when the particles are at the same sites.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a two-dimensional tuple.
        num_variables : int
            The number of variables of this probability distribution.
        state : :py:class:`sparkquantum.dtqw.state.State`
            The quantum state of the system.
        num_elements : int, optional
            The expected (or definitive) number of elements. This helps to find a
            better number of partitions when (re)partitioning the RDD. Default value is None.

        """
        super().__init__(rdd, shape, num_variables, state, num_elements=num_elements)

    def __str__(self):
        return 'Collision Probability Distribution regarding the possible positions of all particles of a {} when the particles are at the same sites'.format(
            self._state)

    def normalize(self, norm=None):
        """Normalize this probability distribution.

        Parameters
        ----------
        norm : float, optional
            The norm of this probability distribution. When `None` is passed as argument, the norm is calculated.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.math.statistics.probability_distribution.position_joint_probability_distribution.PositionJointProbabilityDistribution`
            The normalized probability distribution.

        """
        if norm is None:
            norm = self.norm()

        def __map(m):
            m[-1] /= norm
            return m

        rdd = self.data.map(
            __map
        )

        return PositionJointProbabilityDistribution(
            rdd, self._shape, self._num_variables, self._state, num_elements=self._num_elements
        )
