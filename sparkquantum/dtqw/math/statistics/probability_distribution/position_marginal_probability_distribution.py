from datetime import datetime

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from sparkquantum.dtqw.math.statistics.probability_distribution.position_probability_distribution import PositionProbabilityDistribution

__all__ = ['PositionMarginalProbabilityDistribution']


class PositionMarginalProbabilityDistribution(PositionProbabilityDistribution):
    """Class for marginal probability distributions regarding the possible positions
    of one particle of a quantum system state."""

    def __init__(self, rdd, shape, state, num_elements=None):
        """Build an object for a marginal probability distribution regarding the possible
        positions of one particle of a quantum system state.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a two-dimensional tuple.
        state : :py:class:`sparkquantum.dtqw.state.State`
            The quantum state of the system.
        num_elements : int, optional
            The expected (or definitive) number of elements. This helps to find a
            better number of partitions when (re)partitioning the RDD. Default value is None.

        """
        super().__init__(rdd, shape, 1, state, num_elements=num_elements)

    def __str__(self):
        return 'Marginal Probability Distribution regarding the possible positions of one particle of a {}'.format(
            self._state)

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

        axis = self._state.mesh.axis()

        if self._state.mesh.dimension == 1:
            probability_distribution = np.zeros(self._shape, dtype=float)

            for i in self.data.collect():
                probability_distribution[i[0]] = i[1]

            plt.plot(
                axis,
                probability_distribution,
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
        elif self._state.mesh.dimension == 2:
            probability_distribution = np.zeros(self._shape, dtype=float)

            for i in self.data.collect():
                probability_distribution[i[0], i[1]] = i[2]

            figure = plt.figure()
            axes = figure.add_subplot(111, projection='3d')

            axes.plot_surface(
                axis[0],
                axis[1],
                probability_distribution,
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
            "plot was done in {}s".format(
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
        if self._state.mesh.dimension != 2:
            self._logger.warning(
                "it is only possible to plot the contour function of two-dimensional meshes")
            return None

        self._logger.info("starting plot of probabilities...")

        t1 = datetime.now()

        plt.cla()
        plt.clf()

        axis = self._state.mesh.axis()

        probability_distribution = np.zeros(self._shape, dtype=float)

        for i in self.data.collect():
            probability_distribution[i[0], i[1]] = i[2]

        if 'levels' not in kwargs:
            max_level = probability_distribution.max()

            if not max_level:
                max_level = 1

            levels = np.linspace(0, max_level, 41)

        plt.contourf(
            axis[0],
            axis[1],
            probability_distribution,
            levels=levels,
            **kwargs)
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
            "contour plot was done in {}s".format(
                (datetime.now() - t1).total_seconds()))
