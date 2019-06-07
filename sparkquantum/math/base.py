import numpy as np
from pyspark import RDD, StorageLevel

from sparkquantum.utils.logger import is_logger
from sparkquantum.utils.profiler import is_profiler
from sparkquantum.utils.utils import Utils

__all__ = ['Base']


class Base:
    """Top-level class for some matrix-based elements."""

    def __init__(self, rdd, shape, data_type=complex):
        """Build a top-level object for some matrix-based elements. It is a container of RDD.

        Parameters
        ----------
        rdd : `RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a 2-dimensional tuple.
        data_type : type, optional
            The Python type of all values in this object. Default value is complex.

        """
        if not isinstance(rdd, RDD):
            # self._logger.error("invalid argument to instantiate an RDD-based object")
            raise TypeError("'RDD' instance expected, not '{}'".format(type(rdd)))

        if shape is not None:
            if not Utils.is_shape(shape):
                # self._logger.error("invalid shape")
                raise ValueError("invalid shape")

        self._spark_context = rdd.context
        self._shape = shape
        self._num_elements = self._shape[0] * self._shape[1]
        self._num_nonzero_elements = 0
        self._data_type = data_type

        self.data = rdd

        self._logger = None
        self._profiler = None

    @property
    def spark_context(self):
        """`SparkContext`"""
        return self._spark_context

    @property
    def shape(self):
        """tuple"""
        return self._shape

    @property
    def num_elements(self):
        """int"""
        return self._num_elements

    @property
    def num_nonzero_elements(self):
        """int"""
        return self._num_nonzero_elements

    @property
    def data_type(self):
        """`RDD`"""
        return self._data_type

    @property
    def logger(self):
        """`Logger`.

        To disable logging, set it to `None`.

        """
        return self._logger

    @property
    def profiler(self):
        """`Profiler`.

        To disable profiling, set it to `None`.

        """
        return self._profiler

    @logger.setter
    def logger(self, logger):
        if is_logger(logger) or logger is None:
            self._logger = logger
        else:
            raise TypeError("'Logger' instance expected, not '{}'".format(type(logger)))

    @profiler.setter
    def profiler(self, profiler):
        if is_profiler(profiler) or profiler is None:
            self._profiler = profiler
        else:
            raise TypeError("'Profiler' instance expected, not '{}'".format(type(profiler)))

    def __str__(self):
        return self.__class__.__name__

    def to_string(self):
        return self.__str__()

    def sparsity(self):
        """Calculate the sparsity of this object.

        Returns
        -------
        float
            The sparsity of this object.

        """
        return 1.0 - self.num_nonzero_elements / self._num_elements

    def repartition(self, num_partitions):
        """Changes the number of partitions of this object's RDD.

        Parameters
        ----------
        num_partitions : int
            The target number of partitions of the RDD.

        Returns
        -------
        `self`
            A reference to this object.

        """
        if num_partitions > self.data.getNumPartitions():
            self.data = self.data.repartition(num_partitions)
        elif num_partitions < self.data.getNumPartitions():
            self.data = self.data.coalesce(num_partitions)

        return self

    def define_partitioner(self, num_partitions):
        """Define the hash partitioner with the chosen number of partitions for this object's RDD.

        Parameters
        ----------
        num_partitions : int
            The target number of partitions of the RDD.

        Returns
        -------
        `self`
            A reference to this object.

        """
        self.data = self.data.partitionBy(
            numPartitions=num_partitions
        )

        return self

    def persist(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Persist this object's RDD considering the chosen storage level.

        Parameters
        ----------
        storage_level : `StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is `StorageLevel.MEMORY_AND_DISK`.

        Returns
        -------
        `self`
            A reference to this object.

        """
        if self.data is not None:
            if not self.data.is_cached:
                self.data.persist(storage_level)
                if self._logger:
                    self._logger.info("RDD {} was persisted".format(self.data.id()))
            else:
                if self._logger:
                    self._logger.info("RDD {} has already been persisted".format(self.data.id()))
        else:
            if self._logger:
                self._logger.warning("there is no data to be persisted")

        return self

    def unpersist(self):
        """Unpersist this object's RDD.

        Returns
        -------
        `self`
            A reference to this object.

        """
        if self.data is not None:
            if self.data.is_cached:
                self.data.unpersist()
                if self._logger:
                    self._logger.info("RDD {} was unpersisted".format(self.data.id()))
            else:
                if self._logger:
                    self._logger.info("RDD {} has already been unpersisted".format(self.data.id()))
        else:
            if self._logger:
                self._logger.warning("there is no data to be unpersisted")

        return self

    def destroy(self):
        """Alias of the method unpersist.

        Returns
        -------
        `self`
            A reference to this object.

        """
        return self.unpersist()

    def materialize(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Materialize this object's RDD considering the chosen storage level.

        This method calls persist and right after counts how many elements there are in the RDD to force its
        persistence.

        Parameters
        ----------
        storage_level : `StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is `StorageLevel.MEMORY_AND_DISK`.

        Returns
        -------
        `self`
            A reference to this object.

        """
        self.persist(storage_level)
        self._num_nonzero_elements = self.data.count()

        if self._logger:
            self._logger.info("RDD {} was materialized".format(self.data.id()))

        return self

    def checkpoint(self):
        """Checkpoint this object's RDD.

        Notes
        -----
        If it is intended to use this method in an application, it is necessary to define
        the checkpoint dir using the `SparkContext` object.

        Returns
        -------
        `self`
            A reference to this object.

        """
        if self.data.isCheckpointed():
            if self._logger:
                self._logger.info("RDD already checkpointed")
            return self

        if not self.data.is_cached:
            if self._logger:
                self._logger.warning("it is recommended to cache the RDD before checkpointing it")

        self.data.checkpoint()

        if self._logger:
            self._logger.info("RDD {} was checkpointed in {}".format(self.data.id(), self.data.getCheckpointFile()))

        return self

    def dump(self, path, glue=None, codec=None, filename=None):
        """Dump this object's RDD to disk in a unique file or in many part-* files.

        Notes
        -----
        Depending on the chosen dumping mode, this method calls the RDD's `collect` method.
        This is not suitable for large working sets, as all data may not fit into driver's main memory.

        Parameters
        ----------
        path : str
            The path where the dumped RDD will be located at.
        glue : str, optional
            The glue string that connects each coordinate and value of each element in the RDD.
            Default value is `None`. In this case, it uses the 'quantum.dumpingGlue' configuration value.
        codec : str, optional
            Codec name used to compress the dumped data.
            Default value is `None`. In this case, it uses the 'quantum.dumpingCompressionCodec' configuration value.
        filename : str, optional
            File name used when the dumping mode is in a single file. Default value is None.
            In this case, a temporary named file is generated inside the informed path.

        """
        if glue is None:
            glue = Utils.get_conf(self._spark_context, 'quantum.dumpingGlue')

        if codec is None:
            codec = Utils.get_conf(self._spark_context, 'quantum.dumpingCompressionCodec')

        dumping_mode = int(Utils.get_conf(self._spark_context, 'quantum.math.dumpingMode'))

        if dumping_mode == Utils.DumpingModeUniqueFile:
            data = self.data.collect()

            Utils.create_dir(path)

            if not filename:
                filename = Utils.get_temp_path(path)
            else:
                filename = Utils.append_slash_dir(path) + filename

            if len(data):
                with open(filename, 'a') as f:
                    for d in data:
                        f.write(glue.join([str(e) for e in d]) + "\n")
        elif dumping_mode == Utils.DumpingModePartFiles:
            self.data.map(
                lambda m: glue.join([str(e) for e in m])
            ).saveAsTextFile(path, codec)
        else:
            if self._logger:
                self._logger.error("invalid dumping mode")
            raise NotImplementedError("invalid dumping mode")

    def numpy_array(self):
        """Create a numpy array containing this object's RDD data.

        Notes
        -----
        This method calls the `collect` method of RDD. This is not suitable for large working sets,
        as all data may not fit into main memory.

        Returns
        -------
        ndarray
            The numpy array.

        """
        data = self.data.collect()
        result = np.zeros(self._shape, dtype=self._data_type)

        if len(data):
            ind = len(data[0]) - 1

            for e in data:
                result[e[0:ind]] = e[ind]

        return result
