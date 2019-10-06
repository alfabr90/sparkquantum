import math

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
        mesh : :py:class:`sparkquantum.dtqw.mesh.Mesh`
            The mesh where the particles has walked on.
        num_particles : int
            The number of particles present in the walk.

        """
        super().__init__(rdd, shape, mesh, num_particles)

    def sum_values(self):
        """Sum the probabilities of this PDF.

        Returns
        -------
        float
            The sum of the probabilities.

        Raises
        ------
        NotImplementedError

        """
        if self._mesh.is_1d():
            ind = 1
        elif self._mesh.is_2d():
            ind = 2
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        data_type = self._data_type()

        return self.data.filter(
            lambda m: m[ind] != data_type
        ).map(
            lambda m: m[ind]
        ).reduce(
            lambda a, b: a + b
        )

    def norm(self):
        """Calculate the norm of this PDF.

        Returns
        -------
        float
            The norm of this PDF.

        Raises
        ------
        NotImplementedError

        """
        if self._mesh.is_1d():
            ind = 1
        elif self._mesh.is_2d():
            ind = 2
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        data_type = self._data_type()

        n = self.data.filter(
            lambda m: m[ind] != data_type
        ).map(
            lambda m: m[ind].real ** 2
        ).reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

    def normalize(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Normalize this PDF.

        Parameters
        ----------
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.
            Default value is `StorageLevel.MEMORY_AND_DISK`.

        Returns
        -------
        `CollisionPDF`

        """
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
