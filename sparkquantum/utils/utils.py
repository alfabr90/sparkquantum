import os
import logging
import math
import sys
import tempfile as tf

__all__ = ['Utils']


class Utils():
    """A class that provides utility static methods."""

    DumpingModeUniqueFile = 0
    """
    Indicate that the mathematical entity will be dumped to a unique file in disk. This is not suitable for large working sets,
    as all data must be collected to the driver node of the cluster and may not fit into memory.
    """
    DumpingModePartFiles = 1
    """
    Indicate that the mathematical entity will be dumped to part-* files in disk. This is done in a parallel/distributed way.
    This is the default behaviour.
    """
    MatrixCoordinateDefault = 0
    """
    Indicate that the :py:class:`sparkquantum.math.Matrix` object must have its entries stored as ``(i,j,value)`` coordinates.
    """
    MatrixCoordinateMultiplier = 1
    """
    Indicate that the :py:class:`sparkquantum.math.Matrix` object must have its entries stored as ``(j,(i,value))`` coordinates. This is mandatory
    when the object is the multiplier operand.
    """
    MatrixCoordinateMultiplicand = 2
    """
    Indicate that the :py:class:`sparkquantum.math.Matrix` object must have its entries stored as ``(i,(j,value))`` coordinates. This is mandatory
    when the object is the multiplicand operand.
    """
    StateRepresentationFormatCoinPosition = 0
    """
    Indicate that the quantum system is represented as the kronecker product between the coin and position subspaces.
    """
    StateRepresentationFormatPositionCoin = 1
    """
    Indicate that the quantum system is represented as the kronecker product between the position and coin subspaces.
    """
    StateDumpingFormatIndex = 0
    """
    Indicate that the quantum system will be dumped to disk with the format ``(i,value)``.
    """
    StateDumpingFormatCoordinate = 1
    """
    Indicate that the quantum system will be dumped to disk with the format ``(i1,x1,...,in,xn,value)``,
    for one-dimensional meshes, or ``(i1,j1,x1,y1,...,in,jn,xn,yn,value)``, for two-dimensional meshes,
    for example, where n is the number of particles.
    """
    KroneckerModeBroadcast = 0
    """
    Indicate that the kronecker operation will be performed with the right operand in a broadcast variable.
    Suitable for small working sets, providing the best performance due to all the data be located
    locally in every node of the cluster.
    """
    KroneckerModeDump = 1
    """
    Indicate that the kronecker operation will be performed with the right operand previously dumped to disk.
    Suitable for working sets that do not fit in cache provided by Spark.
    """
    BrokenLinksGenerationModeBroadcast = 0
    """
    Indicate that the broken links/mesh percolation will be generated by a broadcast variable
    containing the edges that are broken. Suitable for small meshes, providing the best performance
    due to all the data be located locally in every node of the cluster.
    """
    BrokenLinksGenerationModeRDD = 1
    """
    Indicate that the broken links/mesh percolation will be generated by a RDD containing the edges that are broken.
    Suitable for meshes that their corresponding data do not fit in the cache provided by Spark. Also, with this mode,
    a left join is performed, making it not the best mode.
    """

    ConfigDefaults = {
        'quantum.cluster.maxPartitionSize': 64 * 10 ** 6,
        'quantum.cluster.numPartitionsSafetyFactor': 1.3,
        'quantum.cluster.totalCores': 1,
        'quantum.cluster.useSparkDefaultNumPartitions': 'False',
        'quantum.dtqw.interactionOperator.checkpoint': 'False',
        'quantum.dtqw.mesh.brokenLinks.generationMode': BrokenLinksGenerationModeBroadcast,
        'quantum.dtqw.profiler.logExecutors': 'False',
        'quantum.dtqw.walk.checkpointingFrequency': -1,
        'quantum.dtqw.walk.checkUnitary': 'False',
        'quantum.dtqw.walk.dumpingFrequency': -1,
        'quantum.dtqw.walk.dumpingPath': './',
        'quantum.dtqw.walk.dumpStatesPDF': 'False',
        'quantum.dtqw.walkOperator.checkpoint': 'False',
        'quantum.dtqw.walkOperator.kroneckerMode': KroneckerModeBroadcast,
        'quantum.dtqw.walkOperator.tempPath': './',
        'quantum.dtqw.state.dumpingFormat': StateDumpingFormatIndex,
        'quantum.dtqw.state.representationFormat': StateRepresentationFormatCoinPosition,
        'quantum.dumpingCompressionCodec': None,
        'quantum.dumpingGlue': ' ',
        'quantum.logging.enabled': 'False',
        'quantum.logging.filename': './log.txt',
        'quantum.logging.format': '%(levelname)s:%(name)s:%(asctime)s:%(message)s',
        'quantum.logging.level': logging.WARNING,
        'quantum.math.dumpingMode': DumpingModePartFiles,
        'quantum.math.roundPrecision': 10,
        'quantum.profiling.enabled': 'False',
        'quantum.profiling.baseUrl': 'http://localhost:4040/api/v1/'
    }
    """
    Dict with the default values for all accepted configurations of the package.
    """

    def __init__(self):
        pass

    @staticmethod
    def is_shape(shape):
        """Check if an object is a shape, i.e., a list or a tuple.

        Parameters
        ----------
        shape : list or tuple
            The object to be checked if it is a shape.

        Returns
        -------
        bool
            True if argument is a shape, False otherwise.

        """
        return isinstance(shape, (list, tuple))

    @staticmethod
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

    @staticmethod
    def get_conf(sc, config_str):
        """Get a configuration value from the :py:class:`pyspark.SparkContext` object.

        Parameters
        ----------
        sc : :py:class:`pyspark.SparkContext`
            The :py:class:`pyspark.SparkContext` object.
        config_str : str
            The configuration string to have its correspondent value obtained.

        Returns
        -------
        str
            The configuration value or None if the configuration is not found.

        """
        c = sc.getConf().get(config_str)

        if not c:
            if config_str not in Utils.ConfigDefaults:
                return None
            return Utils.ConfigDefaults[config_str]

        return c

    @staticmethod
    def change_coordinate(rdd, old_coord, new_coord=MatrixCoordinateDefault):
        """Change the coordinate format of a :py:class:`sparkquantum.math.Matrix` object's RDD.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The :py:class:`sparkquantum.math.Matrix` object's RDD to have its coordinate format changed.
        old_coord : int
            The original coordinate format of the :py:class:`sparkquantum.math.Matrix` object's RDD.
        new_coord : int
            The new coordinate format. Default value is :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault`.

        Returns
        -------
        :py:class:`pyspark.RDD`
            A new :py:class:`pyspark.RDD` with the coordinate format changed.

        """
        if old_coord == Utils.MatrixCoordinateMultiplier:
            if new_coord == Utils.MatrixCoordinateMultiplier:
                return rdd
            elif new_coord == Utils.MatrixCoordinateMultiplicand:
                return rdd.map(
                    lambda m: (m[1][0], (m[0], m[1][1]))
                )
            else:  # Utils.MatrixCoordinateDefault
                return rdd.map(
                    lambda m: (m[1][0], m[0], m[1][1])
                )
        elif old_coord == Utils.MatrixCoordinateMultiplicand:
            if new_coord == Utils.MatrixCoordinateMultiplier:
                return rdd.map(
                    lambda m: (m[1][0], (m[0], m[1][1]))
                )
            elif new_coord == Utils.MatrixCoordinateMultiplicand:
                return rdd
            else:  # Utils.MatrixCoordinateDefault
                return rdd.map(
                    lambda m: (m[0], m[1][0], m[1][1])
                )
        else:  # Utils.MatrixCoordinateDefault
            if new_coord == Utils.MatrixCoordinateMultiplier:
                return rdd.map(
                    lambda m: (m[1], (m[0], m[2]))
                )
            elif new_coord == Utils.MatrixCoordinateMultiplicand:
                return rdd.map(
                    lambda m: (m[0], (m[1], m[2]))
                )
            else:  # Utils.MatrixCoordinateDefault
                return rdd

    @staticmethod
    def get_precedent_type(type1, type2):
        """Compare and return the most precedent type between two types.

        Parameters
        ----------
        type1 : type
            The first type to be compared with.
        type2 : type
            The second type to be compared with.

        Returns
        -------
        type
            The type with most precedent order.

        """
        if type1 == complex or type2 == complex:
            return complex

        if type1 == float or type2 == float:
            return float

        return int

    @staticmethod
    def get_size_of_type(data_type):
        """Get the size in bytes of a Python type.

        Parameters
        ----------
        data_type : type
            The Python type to have its size calculated.

        Returns
        -------
        int
            The size of the Python type in bytes.

        """
        return sys.getsizeof(data_type())

    @staticmethod
    def get_num_partitions(spark_context, expected_size):
        """Calculate the number of partitions for a :py:class:`pyspark.RDD` based on its expected size in bytes.

        Parameters
        ----------
        spark_context : :py:class:`pyspark.SparkContext`
            The :py:class:`pyspark.SparkContext` object.
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
        safety_factor = float(Utils.get_conf(
            spark_context, 'quantum.cluster.numPartitionsSafetyFactor'))
        num_partitions = None

        if Utils.get_conf(
                spark_context, 'quantum.cluster.useSparkDefaultNumPartitions') == 'False':
            num_cores = Utils.get_conf(
                spark_context, 'quantum.cluster.totalCores')

            if not num_cores:
                raise ValueError(
                    "invalid number of total cores in the cluster: {}".format(num_cores))

            num_cores = int(num_cores)
            max_partition_size = int(Utils.get_conf(
                spark_context, 'quantum.cluster.maxPartitionSize'))
            num_partitions = math.ceil(
                safety_factor * expected_size / max_partition_size / num_cores) * num_cores

        return num_partitions

    @staticmethod
    def append_slash_dir(path):
        """Append a slash in a path if it does not end with one.

        Parameters
        ----------
        path : str
            The directory name with its path.

        """
        if not path.endswith('/'):
            path += '/'

        return path

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
                        Utils.remove_path(path + '/' + i)
                    os.rmdir(path)
            else:
                os.remove(path)

    @staticmethod
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
                        Utils.remove_path(path + '/' + i)
            else:
                raise ValueError("'{}' is an invalid path".format(path))

    @staticmethod
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
                size += Utils.get_size_of_path(path + '/' + i)
            return size
        else:
            return os.stat(path).st_size

    @staticmethod
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
            In this case, the value of 'quantum.logging.level' configuration parameter is used.
        filename : int, optional
            The file where the messages will be logged. Default value is `None`.
            In this case, the value of 'quantum.logging.filename' configuration parameter is used.
        format : str, optional
            The log messages format. Default value is `None`.
            In this case, the value of 'quantum.logging.format' configuration parameter is used.

        Returns
        -------
        :py:class:`logging.Logger`
            The :py:class:`logging.Logger` object.

        """
        logger = logging.getLogger(name)

        if level is None:
            level = int(Utils.get_conf(sc, 'quantum.logging.level'))

        logger.setLevel(level)

        if Utils.get_conf(sc, 'quantum.logging.enabled') == 'True':
            if filename is None:
                filename = Utils.get_conf(sc, 'quantum.logging.filename')

            if format is None:
                format = Utils.get_conf(sc, 'quantum.logging.format')

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
