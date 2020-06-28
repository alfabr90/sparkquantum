from sparkquantum.math.statistics.pdf import PDF

__all__ = ['MarginalPDF']


class MarginalPDF(PDF):
    """Class for probability distribution function (PDF) of a particle in the quantum system."""

    def __init__(self, rdd, shape, mesh, num_particles):
        """Build an object for probability distribution function (PDF) of a particle in the quantum system.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a two-dimensional tuple.
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles has walked on.
        num_particles : int
            The number of particles present in the walk.

        """
        super().__init__(rdd, shape, mesh, num_particles)

    def __str__(self):
        if self._num_particles == 1:
            particles = 'one particle'
        else:
            particles = '{} particles'.format(self._num_particles)

        return 'Marginal Probability Distribution Function with shape {} of {} over a {}'.format(
            self._shape, particles, self._mesh)
