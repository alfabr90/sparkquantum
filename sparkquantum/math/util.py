from pyspark import RDD

from sparkquantum import constants

__all__ = ['is_scalar', 'is_shape', 'change_coordinate', 'remove_zeros']


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
    return isinstance(obj, (int, float, complex))


def is_shape(shape):
    """Check if an object is a shape, i.e., a list or a tuple of length 2 and positive values.

    Parameters
    ----------
    shape : list or tuple
        The object to be checked if it is a shape.

    Returns
    -------
    bool
        True if argument is a shape, False otherwise.

    """
    return (isinstance(shape, (list, tuple)) and
            len(shape) == 2 and shape[0] >= 0 and shape[1] >= 0)


def change_coordinate(rdd, old_coordinate, new_coordinate):
    """Change the coordinate format of a :py:class:`sparkquantum.math.matrix.Matrix` object's RDD.

    Notes
    -----
    Due to the immutability of RDD, a new RDD instance is returned. The only exception is when
    `old_coordinate` is equal to `new_coordinate`.

    Parameters
    ----------
    rdd : :py:class:`pyspark.RDD`
        The :py:class:`sparkquantum.math.matrix.Matrix` object's RDD to have its coordinate format changed.
    old_coordinate : int
        The original coordinate format.
    new_coordinate : int
        The new coordinate format.

    Returns
    -------
    :py:class:`pyspark.RDD`
        A new :py:class:`pyspark.RDD` with the coordinate format changed.

    """
    if not isinstance(rdd, RDD):
        raise TypeError("'RDD' instance expected, not '{}'".format(
            type(rdd)))

    if old_coordinate == new_coordinate:
        return rdd

    if old_coordinate != constants.MatrixCoordinateDefault:
        if old_coordinate == constants.MatrixCoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[1][0], m[0], m[1][1])
            )
        elif old_coordinate == constants.MatrixCoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0], m[1][0], m[1][1])
            )
        elif old_coordinate == constants.MatrixCoordinateIndexed:
            rdd = rdd.map(
                lambda m: (m[0][0], m[0][1], m[1])
            )
        else:
            raise ValueError("invalid coordinate format")

    if new_coordinate != constants.MatrixCoordinateDefault:
        if new_coordinate == constants.MatrixCoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[1], (m[0], m[2]))
            )
        elif new_coordinate == constants.MatrixCoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0], (m[1], m[2]))
            )
        elif new_coordinate == constants.MatrixCoordinateIndexed:
            rdd = rdd.map(
                lambda m: ((m[0], m[1]), m[2])
            )
        else:
            raise ValueError("invalid coordinate format")

    return rdd


def remove_zeros(rdd, data_type, coordinate_format):
    """Remove zeros of a :py:class:`sparkquantum.math.matrix.Matrix` object's RDD.

    Notes
    -----
    Due to the immutability of RDD, a new RDD instance is returned.

    Parameters
    ----------
    rdd : :py:class:`pyspark.RDD`
        The :py:class:`sparkquantum.math.matrix.Matrix` object's RDD to have its zeros removed.
    data_type : type
        The Python type of all values in the RDD.
    coordinate_format : int
        The coordinate format of the :py:class:`sparkquantum.math.matrix.Matrix` object's RDD.

    Returns
    -------
    :py:class:`pyspark.RDD`
        A new :py:class:`pyspark.RDD` with the zeros elements removed.

    """
    if not isinstance(rdd, RDD):
        raise TypeError("'RDD' instance expected, not '{}'".format(
            type(rdd)))

    rdd = change_coordinate(
        rdd,
        coordinate_format,
        constants.MatrixCoordinateDefault)

    zero = data_type()

    rdd = rdd.filter(
        lambda m: m[2] != zero
    )

    return change_coordinate(
        rdd,
        constants.MatrixCoordinateDefault,
        coordinate_format)
