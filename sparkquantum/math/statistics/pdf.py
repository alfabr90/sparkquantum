from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.math.base import Base

__all__ = ['PDF', 'is_pdf']


class PDF(Base):
    """Top-level class for probability distribution functions (PDF)."""

    def __init__(self, rdd, shape, mesh, num_particles):
        """Build a top-level object for probability distribution functions (PDF).

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
        super().__init__(rdd, shape, data_type=float)

        self._mesh = mesh
        self._num_particles = num_particles

        if not is_mesh(mesh):
            self._logger.error(
                "'Mesh' instance expected, not '{}'".format(
                    type(mesh)))
            raise TypeError(
                "'Mesh' instance expected, not '{}'".format(
                    type(mesh)))

        if self._num_particles < 1:
            self._logger.error(
                "invalid number of particles. It must be greater than or equal to 1")
            raise ValueError(
                "invalid number of particles. It must be greater than or equal to 1")

    @property
    def mesh(self):
        return self._mesh

    def __str__(self):
        if self._num_particles == 1:
            particles = 'one particle'
        else:
            particles = '{} particles'.format(self._num_particles)

        return 'Probability Distribution Function with shape {} of {} over a {}'.format(
            self._shape, particles, self._mesh)

    def sum_values(self):
        """Sum the values of this PDF.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError

    def norm(self):
        """Calculate the norm of this PDF.

        Raises
        ------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError

    def expected_value(self):
        """Calculate the expected value of this PDF.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError

    def variance(self, mean=None):
        """Calculate the variance of this PDF.

        Parameters
        ----------
        mean : float, optional
            The mean of this PDF. When None is passed as argument, the mean is calculated.

        Raises
        ------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError

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
        if self._mesh.dimension == 1:
            ind = 1
        elif self._mesh.dimension == 2:
            ind = 2
        else:
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
        if self._mesh.dimension == 1:
            ind = 1
        elif self._mesh.dimension == 2:
            ind = 2
        else:
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

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        """
        self._logger.info("starting plot of probabilities...")

        t1 = datetime.now()

        plt.cla()
        plt.clf()

        axis = self._mesh.axis()

        if self._mesh.dimension == 1:
            pdf = np.zeros(self._shape, dtype=float)

            for i in self.data.collect():
                pdf[i[0]] = i[1]

            plt.plot(
                axis,
                pdf,
                color='b',
                linestyle='-',
                linewidth=1.0
            )

            if labels:
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
            else:
                plt.xlabel("Position")
                plt.ylabel("Probability")

            if title:
                plt.title(title)
        elif self._mesh.dimension == 2:
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
                axes.set_xlabel("Position x")
                axes.set_ylabel("Position y")
                axes.set_zlabel("Probability")

            if title:
                axes.set_title(title)
            axes.view_init(elev=50)

            # figure.set_size_inches(12.8, 12.8)
        else:
            self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        plt.savefig(filename, **kwargs)
        plt.cla()
        plt.clf()

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
        if self._mesh.dimension != 2:
            self._logger.warning(
                "it is only possible to plot the contour function of two-dimensional meshes")
            return None

        self._logger.info("starting plot of probabilities...")

        t1 = datetime.now()

        plt.cla()
        plt.clf()

        axis = self._mesh.axis()

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
            plt.xlabel("Position x")
            plt.ylabel("Position y")

        if title:
            plt.title(title)

        # figure.set_size_inches(12.8, 12.8)

        plt.savefig(filename, **kwargs)
        plt.cla()
        plt.clf()

        self._logger.info(
            "contour plot in {}s".format(
                (datetime.now() - t1).total_seconds()))


def is_pdf(obj):
    """Check whether argument is a :py:class:`sparkquantum.math.statistics.pdf.PDF` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.math.statistics.pdf.PDF` object, False otherwise.

    """
    return isinstance(obj, PDF)
