from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkContext

__all__ = ['line', 'surface', 'contour']


def line(axis, data, filename, title=None, labels=None, **kwargs):
    """Plot a line graph.

    Parameters
    ----------
    axis: array-like
        The values of the x coordinate.
    data: array-like
        The values of the y coordinate.
    filename: str
        The filename to save the plot.
    title: str, optional
        The title of the plot.
    labels: tuple, optional
        The labels of each axis.
    \*\*kwargs
        Keyword arguments being passed to `matplotlib <https://matplotlib.org>`_.

    """
    plt.cla()
    plt.clf()

    plt.plot(
        axis,
        data,
        color='b',
        linestyle='-',
        linewidth=1.0
    )

    if labels is not None:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    if title is not None:
        plt.title(title)

    plt.savefig(filename, **kwargs)
    plt.cla()
    plt.clf()


def surface(axis, data, filename, title=None, labels=None, **kwargs):
    """Plot a surface graph.

    Parameters
    ----------
    axis: array-like
        The values of the x and y coordinates.
    data: array-like
        The values of the z coordinate.
    filename: str
        The filename to save the plot.
    title: str, optional
        The title of the plot.
    labels: tuple, optional
        The labels of each axis.
    \*\*kwargs
        Keyword arguments being passed to `matplotlib <https://matplotlib.org>`_.

    """
    plt.cla()
    plt.clf()

    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')

    meshgrid = np.meshgrid(axis[0], axis[1], indexing='ij')

    axes.plot_surface(
        meshgrid[0],
        meshgrid[1],
        data,
        rstride=1,
        cstride=1,
        cmap=cm.YlGnBu_r,
        linewidth=0.1,
        antialiased=True
    )

    if labels is not None:
        axes.set_xlabel(labels[0])
        axes.set_ylabel(labels[1])
        axes.set_zlabel(labels[2])

    if title is not None:
        axes.set_title(title)

    axes.view_init(elev=50)

    # figure.set_size_inches(12.8, 12.8)

    plt.savefig(filename, **kwargs)
    plt.cla()
    plt.clf()


def contour(axis, data, filename, title=None, labels=None, **kwargs):
    """Plot a contour graph.

    Parameters
    ----------
    axis: array-like
        The values of the x and y coordinates.
    data: array-like
        The values of the z coordinate.
    filename: str
        The filename to save the plot.
    title: str, optional
        The title of the plot.
    labels: tuple, optional
        The labels of each axis.
    \*\*kwargs
        Keyword arguments being passed to `matplotlib <https://matplotlib.org>`_.

    """
    plt.cla()
    plt.clf()

    if 'levels' not in kwargs:
        max_level = data.max()

        if not max_level:
            max_level = 1

        levels = np.linspace(0, max_level, 41)

    plt.contourf(
        axis[0],
        axis[1],
        data,
        levels=levels,
        **kwargs)

    plt.colorbar()

    if labels is not None:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    if title is not None:
        plt.title(title)

    # figure.set_size_inches(12.8, 12.8)

    plt.savefig(filename, **kwargs)
    plt.cla()
    plt.clf()
