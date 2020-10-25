import os
import logging
import math
import sys
import tempfile as tf
from distutils.util import strtobool

from pyspark import RDD

from sparkquantum import conf

__all__ = [
    'broadcast',
    'get_precedent_type',
    'get_size_of_type',
    'get_num_partitions',
    'append_slash',
    'create_dir',
    'get_temp_path',
    'remove_path',
    'clear_path',
    'get_size_of_path',
    'get_logger']


def to_bool(val):
    """Convert a value to true or false.

    Parameters
    ----------
    val
        The value to be converted.

    Returns
    -------
    bool

    """
    if val is None:
        return False
    elif isinstance(val, bool):
        return val
    elif isinstance(val, str):
        return bool(strtobool(val))
    else:
        return bool(val)


def broadcast(sc, data):
    """Broadcast some data.

    Parameters
    ----------
    sc : :py:class:`pyspark.SparkContext`
        The :py:class:`pyspark.SparkContext` object.
    data
        The data to be broadcast.

    Returns
    -------
    :py:class:`pyspark.Broadcast`
        The :py:class:`pyspark.Broadcast` object that contains the broadcast data.

    """
    return sc.broadcast(data)


def get_precedent_type(type1, type2):
    """Compare and return the most precedent type between two numeric
    types, i.e., int, float or complex.

    Parameters
    ----------
    type1 : type
        The first type to be compared with.
    type2 : type
        The second type to be compared with.

    Returns
    -------
    type
        The numeric type with most precedent order.

    """
    if type1 == complex or type2 == complex:
        return complex

    if type1 == float or type2 == float:
        return float

    return int


def get_size_of_type(dtype):
    """Get the size in bytes of a Python type.

    Parameters
    ----------
    dtype : type
        The Python type to have its size calculated.

    Returns
    -------
    int
        The size of the Python type in bytes.

    """
    return sys.getsizeof(dtype())


def get_num_partitions(spark_context, expected_size):
    """Calculate the number of partitions for a :py:class:`pyspark.RDD` based on its expected size in bytes.

    Parameters
    ----------
    spark_context : :py:class:`pyspark.SparkContext`confighe :py:class:`pyspark.SparkContext` object.
    expected_size : int
        The expected size in bytes of the RDD.

    Returns
    -------
    int
        The number of partitions for the RDD.

    Raises
    ------
    ValueError
        If the number of cores is not informed.

    """
    safety_factor = float(conf.get(
        spark_context, 'sparkquantum.cluster.numPartitionsSafetyFactor'))

    num_partitions = None

    if not to_bool(conf.get(spark_context,
                            'sparkquantum.cluster.useSparkDefaultNumPartitions')):
        num_cores = conf.get(
            spark_context, 'sparkquantum.cluster.totalCores')

        if not num_cores:
            raise ValueError(
                "invalid number of total cores in the cluster: {}".format(num_cores))

        num_cores = int(num_cores)
        max_partition_size = int(conf.get(
            spark_context, 'sparkquantum.cluster.maxPartitionSize'))
        num_partitions = math.ceil(
            safety_factor * expected_size / max_partition_size / num_cores) * num_cores

    return num_partitions


def append_slash(path):
    """Append a slash in a path if it does not end with one.

    Parameters
    ----------
    path : str
        The directory name with its path.

    """
    if not path.endswith('/'):
        path += '/'

    return path


def create_dir(path):
    """Create a directory in the filesystem.

    Parameters
    ----------
    path : str
        The directory name with its path.

    Raises
    ------
    ValueError
        If `path` is not valid.

    """
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError("'{}' is an invalid path".format(path))
    else:
        os.makedirs(path)


def get_temp_path(d):
    """Create a temp directory in the filesystem.

    Parameters
    ----------
    d : str
        The name of the temp path.

    """
    tmp_file = tf.NamedTemporaryFile(dir=d)
    tmp_file.close()

    return tmp_file.name


def remove_path(path):
    """Delete a directory in the filesystem.

    Parameters
    ----------
    path : str
        The directory name with its path.

    """
    if os.path.exists(path):
        if os.path.isdir(path):
            if path != '/':
                for i in os.listdir(path):
                    remove_path(path + '/' + i)
                os.rmdir(path)
        else:
            os.remove(path)


def clear_path(path):
    """Empty a directory in the filesystem.

    Parameters
    ----------
    path : str
        The directory name with its path.

    Raises
    ------
    ValueError
        If `path` is not valid.

    """
    if os.path.exists(path):
        if os.path.isdir(path):
            if path != '/':
                for i in os.listdir(path):
                    remove_path(path + '/' + i)
        else:
            raise ValueError("'{}' is an invalid path".format(path))


def get_size_of_path(path):
    """Get the size in bytes of a directory in the filesystem.

    Parameters
    ----------
    path : str
        The directory name with its path.

    Returns
    -------
    int
        The size in bytes of the directory.

    """
    if os.path.isdir(path):
        size = 0
        for i in os.listdir(path):
            size += get_size_of_path(path + '/' + i)
        return size
    else:
        return os.stat(path).st_size


def get_logger(
        sc, name, level=None, filename=None, format=None):
    """Create a :py:class:`logging.Logger` object.

    Parameters
    ----------
    sc : :py:class:`pyspark.SparkContext`
        The :py:class:`pyspark.SparkContext` object.
    name : str
        The name of the class that is providing log data.
    level : int, optional
        The log level. Default value is `None`.
        In this case, the value of 'sparkquantum.logging.level' configuration parameter is used.
    filename : int, optional
        The file where the messages will be logged. Default value is `None`.
        In this case, the value of 'sparkquantum.logging.filename' configuration parameter is used.
    format : str, optional
        The log messages format. Default value is `None`.
        In this case, the value of 'sparkquantum.logging.format' configuration parameter is used.

    Returns
    -------
    :py:class:`logging.Logger`
        The :py:class:`logging.Logger` object.

    """
    logger = logging.getLogger(name)

    if level is None:
        level = int(conf.get(sc, 'sparkquantum.logging.level'))

    logger.setLevel(level)

    if to_bool(conf.get(sc, 'sparkquantum.logging.enabled')):
        if filename is None:
            filename = conf.get(sc, 'sparkquantum.logging.filename')

        if format is None:
            format = conf.get(sc, 'sparkquantum.logging.format')

        formatter = logging.Formatter(format)

        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        found_file_handler = False

        # As :py:func:`logging.getLogger` can return an already existent logger,
        # i.e, a previously created logger with the same name,
        # we need to ensure that the logger does not already have a FileHandler
        # that writes to the same location of the FileHandler that is up to
        # be added.
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                if h.baseFilename == file_handler.baseFilename:
                    found_file_handler = True

        if not found_file_handler:
            logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler(level))

    return logger
