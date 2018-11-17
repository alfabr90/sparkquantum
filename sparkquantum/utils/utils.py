import os
import sys
import math
import tempfile as tf

__all__ = ['Utils']


class Utils():
    """A class that provides utility static methods."""

    CoordinateDefault = 0
    """
    CoordinateDefault : int
        Indicate that the `Matrix` object must have its entries stored as (i,j,value) coordinates.
    """
    CoordinateMultiplier = 1
    """
    CoordinateMultiplier : int
        Indicate that the `Matrix` object must have its entries stored as (j,(i,value)) coordinates. This is mandatory
        when the object is the multiplier operand.
    """
    CoordinateMultiplicand = 2
    """
    CoordinateMultiplicand : int
        Indicate that the `Matrix` object must have its entries stored as (i,(j,value)) coordinates. This is mandatory
        when the object is the multiplicand operand.
    """
    RepresentationFormatCoinPosition = 0
    """
    RepresentationFormatCoinPosition : int
        Indicate that the quantum system is represented as the kronecker product between the coin and position subspaces.
    """
    RepresentationFormatPositionCoin = 1
    """
    RepresentationFormatPositionCoin : int
        Indicate that the quantum system is represented as the kronecker product between the position and coin subspaces.
    """

    def __init__(self):
        pass

    @staticmethod
    def is_shape(shape):
        """Check if an object is a shape, i.e., a list or a tuple.

        Parameters
        ----------
        shape :
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
        sc : `SparkContext`
            The `SparkContext` object.
        data :
            The data to be broadcast.

        Returns
        -------
        `Broadcast`
            The `Broadcast` object that contains the broadcast data.

        """
        return sc.broadcast(data)

    @staticmethod
    def get_conf(sc, config_str, default=None):
        """Get a configuration value from the `SparkContext` object.

        Parameters
        ----------
        sc : `SparkContext`
            The `SparkContext` object.
        config_str : str
            The configuration string to have its correspondent value obtained.
        default :
            The default value of the configuration string. Default value is `None`.

        Returns
        -------
        str
            The configuration value.

        """
        c = sc.getConf().get(config_str)

        if not c and (default is not None):
            return default

        return c

    @staticmethod
    def change_coordinate(rdd, old_coord, new_coord=CoordinateDefault):
        """Change the coordinate format of a `Matrix` object's RDD.

        Parameters
        ----------
        rdd : `RDD`
            The `Matrix` object's RDD to have its coordinate format changed.
        old_coord : int
            The original coordinate format of the `Matrix` object's RDD.
        new_coord : int
            The new coordinate format. Default value is `Utils.CoordinateDefault`.

        Returns
        -------
        `RDD`
            A new `RDD` with the coordinate format changed.

        """
        if old_coord == Utils.CoordinateMultiplier:
            if new_coord == Utils.CoordinateMultiplier:
                return rdd
            elif new_coord == Utils.CoordinateMultiplicand:
                return rdd.map(
                    lambda m: (m[1][0], (m[0], m[1][1]))
                )
            else:  # Utils.CoordinateDefault
                return rdd.map(
                    lambda m: (m[1][0], m[0], m[1][1])
                )
        elif old_coord == Utils.CoordinateMultiplicand:
            if new_coord == Utils.CoordinateMultiplier:
                return rdd.map(
                    lambda m: (m[1][0], (m[0], m[1][1]))
                )
            elif new_coord == Utils.CoordinateMultiplicand:
                return rdd
            else:  # Utils.CoordinateDefault
                return rdd.map(
                    lambda m: (m[0], m[1][0], m[1][1])
                )
        else:  # Utils.CoordinateDefault
            if new_coord == Utils.CoordinateMultiplier:
                return rdd.map(
                    lambda m: (m[1], (m[0], m[2]))
                )
            elif new_coord == Utils.CoordinateMultiplicand:
                return rdd.map(
                    lambda m: (m[0], (m[1], m[2]))
                )
            else:  # Utils.CoordinateDefault
                return rdd

    @staticmethod
    def filename(mesh_filename, steps, num_particles):
        """Build a filename concatenating the parameters.

        Parameters
        ----------
        mesh_filename : str
            The generated name for the used mesh.
        steps : int
            The number of steps of the walk.
        num_particles : int
            The number of particles in the walk.

        Returns
        -------
        str
            The filename built.

        """
        return "{}_{}_{}".format(mesh_filename, steps, num_particles)

    @staticmethod
    def get_precendent_type(type1, type2):
        """Compare and return the most precendent type between two types.

        Parameters
        ----------
        type1 : type
            The first type to be compared with.
        type2 : type
            The second type to be compared with.

        Returns
        -------
        type
            The type with most precendent order.

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
        """Calculate the number of partitions for a `RDD` based on its expected size in bytes.

        Parameters
        ----------
        spark_context : `SparkContext`
            The `SparkContext` object.
        expected_size : int
            The expected size in bytes of the RDD.

        Returns
        -------
        int
            The number of partitions for the RDD.

        Raises
        ------
        `ValueError`

        """
        safety_factor = float(Utils.get_conf(spark_context, 'quantum.cluster.numPartitionsSafetyFactor', default=1.3))
        num_partitions = None

        if Utils.get_conf(spark_context, 'quantum.dtqw.useSparkDefaultPartitions', default='False') == 'False':
            num_cores = Utils.get_conf(spark_context, 'quantum.cluster.totalCores', default=None)

            if not num_cores:
                raise ValueError("invalid number of total cores in the cluster: {}".format(num_cores))

            num_cores = int(num_cores)
            max_partition_size = int(Utils.get_conf(spark_context, 'quantum.cluster.maxPartitionSize', default=64 * 10 ** 6))
            num_partitions = math.ceil(safety_factor * expected_size / max_partition_size / num_cores) * num_cores

        return num_partitions

    @staticmethod
    def create_dir(path):
        """Create a directory in the filesystem.

        Parameters
        ----------
        path : str
            The directory name with its path.

        Raises
        ------
        `ValueError`

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

        Raises
        ------
        `ValueError`

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
        `ValueError`

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
