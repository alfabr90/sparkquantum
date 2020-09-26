from pyspark import RDD

from sparkquantum import constants

__all__ = ['is_scalar', 'is_shape']


def is_scalar(obj):
    """Check if an object is a scalar (number), i.e., an int, a float or a complex.

    Parameters
    ----------
    obj
        Any python object.

    Returns
    -------
    bool
        True if argument is a scalar, False otherwise.

    """
    return not isinstance(obj, bool) and isinstance(obj, (int, float, complex))


def is_shape(obj, ndim=None):
    """Check if an object is a shape, i.e., a n-dimension (n positive) tuple with positive values.

    Parameters
    ----------
    obj : tuple
        The object to be checked if it is a shape.
    ndim : int, optional
        The expected number of dimensions. Default value is None.

    Returns
    -------
    bool
        True if argument is a shape, False otherwise.

    """
    if not isinstance(obj, tuple):
        return False

    if len(obj) == 0:
        return False

    if ndim is not None and len(obj) != ndim:
        return False

    for n in obj:
        if n <= 0:
            return False

    return True
