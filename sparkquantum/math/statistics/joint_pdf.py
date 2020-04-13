import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

from sparkquantum.math.statistics.pdf import PDF

__all__ = ['JointPDF']


class JointPDF(PDF):
    """Class for probability distribution function (PDF) of the entire quantum system."""

    def __init__(self, rdd, shape, mesh, num_particles):
        """Build an object for probability distribution function (PDF) of the entire quantum system.

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

        return 'Joint Probability Distribution Function with shape {} of {} over a {}'.format(
            self._shape, particles, self._mesh.to_string())

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
            ind = self._num_particles
        elif self._mesh.is_2d():
            ind = self._num_particles * 2
        else:
            if self._logger is not None:
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
            ind = self._num_particles
        elif self._mesh.is_2d():
            ind = 2 * self._num_particles
        else:
            if self._logger is not None:
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
            ndim = 1
            ind = ndim * self._num_particles
            mesh_size = (int(self._mesh.size / 2), 1)
        elif self._mesh.is_2d():
            ndim = 2
            ind = ndim * self._num_particles
            mesh_size = (
                int(self._mesh.size[0] / 2), int(self._mesh.size[1] / 2))
        else:
            if self._logger is not None:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        def _map(m):
            v = 1

            for i in range(0, ind, ndim):
                for d in range(ndim):
                    v *= m[i + d] - mesh_size[d]

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
            ndim = 1
            ind = ndim * self._num_particles
            mesh_size = (int(self._mesh.size / 2), 1)
        elif self._mesh.is_2d():
            ndim = 2
            ind = ndim * self._num_particles
            mesh_size = (
                int(self._mesh.size[0] / 2), int(self._mesh.size[1] / 2))
        else:
            if self._logger is not None:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        if mean is None:
            mean = self.expected_value()

        def _map(m):
            v = 1

            for i in range(0, ind, ndim):
                for d in range(ndim):
                    v *= m[i + d] - mesh_size[d]

            return m[ind] * v ** 2

        data_type = self._data_type()

        return self.data.filter(
            lambda m: m[ind] != data_type
        ).map(
            _map
        ).reduce(
            lambda a, b: a + b
        ) - mean

    def max(self):
        """Find the maximum value of this PDF.

        Returns
        ------
        float
            The maximum value of this PDF.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        """
        if self._mesh.is_1d():
            ind = self._num_particles
        elif self._mesh.is_2d():
            ndim = 2
            ind = ndim * self._num_particles
        else:
            if self._logger is not None:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        def __map(m):
            return m[ind]

        return self.data.map(
            __map
        ).max()

    def min(self):
        """Find the minimum value of this PDF.

        Returns
        ------
        float
            The minimum value of this PDF.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        """
        if self._mesh.is_1d():
            ind = self._num_particles
        elif self._mesh.is_2d():
            ndim = 2
            ind = ndim * self._num_particles
        else:
            if self._logger is not None:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        def __map(m):
            return m[ind]

        return self.data.map(
            __map
        ).min()

    def plot(self, filename, title=None, labels=None, **kwargs):
        """Plot the probabilities over the mesh.

        Parameters
        ----------
        filename: str
            The filename to save the plot.
        title: str, optional
            The title of the plot.
        labels: tuple or list, optional
            The labels of each axis.
        \*\*kwargs
            Keyword arguments being passed to `matplotlib <https://matplotlib.org>`_.

        """
        if self._mesh.is_1d() and self._num_particles > 2:
            if self._logger is not None:
                self._logger.warning("for one-dimensional meshes, \
                    it is only possible to plot the joint probabilities \
                    of systems of one and two particles"
                                     )
            return None

        if self._mesh.is_2d() and self._num_particles > 1:
            if self._logger is not None:
                self._logger.warning("for two-dimensional meshes, \
                    it is only possible to plot the joint probabilities \
                    of systems of just one particle"
                                     )
            return None

        if not (self._mesh.is_1d() and self._num_particles == 2):
            super().plot(filename, title=title, labels=labels, **kwargs)
        else:
            if self._logger is not None:
                self._logger.info("starting plot of probabilities...")

            t1 = datetime.now()

            plt.cla()
            plt.clf()

            axis = np.meshgrid(
                self._mesh.axis(),
                self._mesh.axis(),
                indexing='ij'
            )

            pdf = np.zeros(self._shape, dtype=float)

            for i in self.data.collect():
                pdf[i[0], i[1]] = i[2]

            figure = plt.figure()
            axes = figure.add_subplot(111, projection='3d')

            axes.plot_surface(
                axis[0],
                axis[1],
                pdf,
                rstride=1,
                cstride=1,
                cmap=cm.YlGnBu_r,
                linewidth=0.1,
                antialiased=True
            )

            if labels:
                axes.set_xlabel(labels[0])
                axes.set_ylabel(labels[1])
                axes.set_zlabel(labels[2])
            else:
                axes.set_xlabel("Particle 1")
                axes.set_ylabel("Particle 2")
                axes.set_zlabel("Probability")

            if title:
                axes.set_title(title)
            axes.view_init(elev=50)

            # figure.set_size_inches(12.8, 12.8)

            plt.savefig(filename, **kwargs)
            plt.cla()
            plt.clf()

            if self._logger is not None:
                self._logger.info(
                    "plot in {}s".format(
                        (datetime.now() - t1).total_seconds()))

    def plot_contour(self, filename=None, title=None, labels=None, **kwargs):
        """Plot the contour function of the probabilities over the mesh.

        Parameters
        ----------
        filename: str
            The filename to save the plot.
        title: str, optional
            The title of the plot.
        labels: tuple or list, optional
            The labels of each axis.
        \*\*kwargs
            Keyword arguments being passed to `matplotlib <https://matplotlib.org>`_.

        """
        if self._mesh.is_1d() and self._num_particles != 2:
            if self._logger is not None:
                self._logger.warning("for one-dimensional meshes, \
                    it is only possible to plot the contour of the joint probability \
                    of systems of two particles"
                                     )
            return None

        if self._mesh.is_2d() and self._num_particles > 1:
            if self._logger is not None:
                self._logger.warning("for two-dimensional meshes, \
                    it is only possible to plot the contour of the joint probability \
                    of systems of just one particle"
                                     )
            return None

        if not (self._mesh.is_1d() and self._num_particles == 2):
            super().plot_contour(filename, title=title, labels=labels, **kwargs)
        else:
            if self._logger is not None:
                self._logger.info("starting contour plot of probabilities...")

            t1 = datetime.now()

            plt.cla()
            plt.clf()

            axis = np.meshgrid(
                self._mesh.axis(),
                self._mesh.axis(),
                indexing='ij'
            )

            pdf = np.zeros(self._shape, dtype=float)

            for i in self.data.collect():
                pdf[i[0], i[1]] = i[2]

            if 'levels' not in kwargs:
                max_level = pdf.max()

                if not max_level:
                    max_level = 1

                levels = np.linspace(0, max_level, 41)

            plt.contourf(axis[0], axis[1], pdf, levels=levels, **kwargs)
            plt.colorbar()

            if labels:
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
            else:
                plt.xlabel("Particle 1")
                plt.ylabel("Particle 2")

            if title:
                plt.title(title)

            # figure.set_size_inches(12.8, 12.8)

            plt.savefig(filename, **kwargs)
            plt.cla()
            plt.clf()

            if self._logger is not None:
                self._logger.info(
                    "contour plot in {}s".format(
                        (datetime.now() - t1).total_seconds()))
