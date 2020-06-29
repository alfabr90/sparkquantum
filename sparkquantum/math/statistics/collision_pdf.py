from pyspark import StorageLevel

from sparkquantum.math.statistics.pdf import PDF

__all__ = ['CollisionPDF']


class CollisionPDF(PDF):
    """Class for probability distribution function (PDF) of the quantum system when the particles are at the same sites."""

    def __init__(self, rdd, shape, mesh, num_particles):
        """
        Build an object for probability distribution function (PDF) of the quantum system when the particles are at the same sites.

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

        return 'Collistion Probability Distribution Function of {} with shape {} over a {}'.format(
            particles, self._shape, self._mesh)

    def normalize(self, norm=None, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Normalize this PDF.

        Notes
        -----
        The RDD of the normalized PDF is already materialized.

        Parameters
        ----------
        norm : float, optional
            The norm of this PDF. When `None` is passed as argument, the norm is calculated.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.
            Default value is :py:const:`pyspark.StorageLevel.MEMORY_AND_DISK`.

        Returns
        -------
        :py:class:`sparkquantum.math.statistics.collision_pdf.CollisionPDF`
            The normalized PDF.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        """
        if norm is None:
            norm = self.norm()

        def __map(m):
            m[-1] /= norm
            return m

        rdd = self.data.map(
            __map
        )

        return CollisionPDF(
            rdd, self._shape, self._mesh, self._num_particles
        ).materialize(storage_level)
