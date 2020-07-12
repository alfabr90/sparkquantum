from datetime import datetime

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from sparkquantum.dtqw.math.statistics.probability_distribution.position_probability_distribution import PositionProbabilityDistribution

__all__ = ['PositionJointProbabilityDistribution']


class PositionJointProbabilityDistribution(PositionProbabilityDistribution):
    """Class for joint probability distributions regarding the possible positions
    of all particles of a quantum system state."""

    def __init__(self, rdd, shape, num_variables, state, num_elements=None):
        """Build an object for a joint probability distribution regarding the possible
        positions of all particles of a quantum system state.

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
        return 'Joint Probability Distribution regarding the possible positions of all particles of a {}'.format(
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

        """
        if self._state.mesh.dimension == 1 and self._state.num_particles > 2:
            self._logger.warning("for one-dimensional meshes, \
                    it is only possible to plot the joint probabilities \
                    of systems of one and two particles"
                                 )
            return None

        if self._state.mesh.dimension == 2 and self._state.num_particles > 1:
            self._logger.warning("for two-dimensional meshes, \
                    it is only possible to plot the joint probabilities \
                    of systems of just one particle"
                                 )
            return None

        if not (self._state.mesh.dimension ==
                1 and self._state.num_particles == 2):
            super().plot(filename, title=title, labels=labels, **kwargs)
        else:
            self._logger.info("starting plot of probabilities...")

            t1 = datetime.now()

            plt.cla()
            plt.clf()

            axis = np.meshgrid(
                self._state.mesh.axis(),
                self._state.mesh.axis(),
                indexing='ij'
            )

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
        if self._state.mesh.dimension == 1 and self._state.num_particles != 2:
            self._logger.warning("for one-dimensional meshes, \
                    it is only possible to plot the contour of the joint probability \
                    of systems of two particles"
                                 )
            return None

        if self._state.mesh.dimension == 2 and self._state.num_particles > 1:
            self._logger.warning("for two-dimensional meshes, \
                    it is only possible to plot the contour of the joint probability \
                    of systems of just one particle"
                                 )
            return None

        if not (self._state.mesh.dimension ==
                1 and self._state.num_particles == 2):
            super().plot_contour(filename, title=title, labels=labels, **kwargs)
        else:
            self._logger.info("starting contour plot of probabilities...")

            t1 = datetime.now()

            plt.cla()
            plt.clf()

            axis = np.meshgrid(
                self._state.mesh.axis(),
                self._state.mesh.axis(),
                indexing='ij'
            )

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
                plt.xlabel("Particle 1")
                plt.ylabel("Particle 2")

            if title:
                plt.title(title)

            # figure.set_size_inches(12.8, 12.8)

            plt.savefig(filename, **kwargs)
            plt.cla()
            plt.clf()

            self._logger.info(
                "contour plot was done in {}s".format(
                    (datetime.now() - t1).total_seconds()))
