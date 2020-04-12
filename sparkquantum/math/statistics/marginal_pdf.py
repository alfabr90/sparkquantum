import math

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

    def sum_values(self):
        """Sum the values of this PDF.

        Returns
        -------
        float
            The sum of the probabilities.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

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
            If the dimension of the mesh is not valid.

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

    def expected_value(self):
        """Calculate the expected value of this PDF.

        Returns
        -------
        float
            The expected value of this PDF.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        """
        if self._mesh.is_1d():
            ind = 1
            mesh_size = (int(self._mesh.size / 2), 1)
        elif self._mesh.is_2d():
            ind = 2
            mesh_size = (
                int(self._mesh.size[0] / 2), int(self._mesh.size[1] / 2))
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        def _map(m):
            v = 1

            for i in range(ind):
                v *= m[i] - mesh_size[i]

            return m[ind] * v

        data_type = self._data_type()

        return self.data.filter(
            lambda m: m[ind] != data_type
        ).map(
            _map
        ).reduce(
            lambda a, b: a + b
        )

    def variance(self, mean=None):
        """Calculate the variance of this PDF.

        Parameters
        ----------
        mean : float, optional
            The mean of this PDF. When None is passed as argument, the mean is calculated.

        Returns
        -------
        float
            The variance of this PDF.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        """
        if self._mesh.is_1d():
            ind = 1
            mesh_size = (int(self._mesh.size / 2), 1)
        elif self._mesh.is_2d():
            ind = 2
            mesh_size = (
                int(self._mesh.size[0] / 2), int(self._mesh.size[1] / 2))
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        if mean is None:
            mean = self.expected_value()

        def _map(m):
            v = 1

            for i in range(ind):
                v *= m[i] - mesh_size[i]

            return m[ind] * v ** 2

        data_type = self._data_type()

        return self.data.filter(
            lambda m: m[ind] != data_type
        ).map(
            _map
        ).reduce(
            lambda a, b: a + b
        ) - mean
