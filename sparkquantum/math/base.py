import numpy as np

from pyspark import StorageLevel
from pyspark.sql import DataFrame

from sparkquantum.utils.logger import is_logger
from sparkquantum.utils.profiler import is_profiler
from sparkquantum.utils.utils import Utils

__all__ = ['Base']


class Base:
    """Top-level class for some matrix-based elements."""

    def __init__(self, df, shape, data_type=complex):
        """Build a top-level object for some matrix-based elements. It is a container of DataFrame.

        Parameters
        ----------
        df : `DataFrame`
            The base DataFrame of this object.
        shape : tuple
            The shape of this matrix object. Must be a 2-dimensional tuple.
        data_type : type, optional
            The Python type of all values in this object. Default value is complex.

        """
        if not isinstance(df, DataFrame):
            # self._logger.error("'DataFrame' instance expected, not '{}'".format(type(df)))
            raise TypeError("'DataFrame' instance expected, not '{}'".format(type(df)))

        if shape is not None:
            if not Utils.is_shape(shape):
                # self._logger.error("invalid shape")
                raise ValueError("invalid shape")

        self._spark_session = df.sql_ctx.sparkSession
        self._shape = shape
        self._num_elements = self._shape[0] * self._shape[1]
        self._num_nonzero_elements = 0
        self._data_type = data_type
        self._is_checkpointed = False

        self.data = df
        self._fields = self.data.schema.names

        self._logger = None
        self._profiler = None

    @property
    def spark_session(self):
        """`SparkSession`"""
        return self._spark_session

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
        """`DataFrame`"""
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

    def rdd(self):
        """Return the corresponding `RDD` of this object's `DataFrame`."""
        return self.data.rdd

    def isCached(self):
        """Return if this object's `DataFrame` is cached."""
        return self.data.is_cached

    def isCheckpointed(self):
        """Return if this object's `DataFrame` is checkpointed."""
        return self._is_checkpointed

    def sparsity(self):
        """Calculate the sparsity of this object.

        Returns
        -------
        float
            The sparsity of this object.

        """
        return 1.0 - self.num_nonzero_elements / self._num_elements

    def repartition(self, num_partitions):
        """Change the number of partitions of this object's DataFrame.

        Parameters
        ----------
        num_partitions : int
            The target number of partitions of the DataFrame.

        Returns
        -------
        `self`
            A reference to this object.

        """
        if num_partitions > self.data.rdd.getNumPartitions():
            self.data = self.data.rdd.repartition(num_partitions)
        elif num_partitions < self.data.rdd.getNumPartitions():
            self.data = self.data.rdd.coalesce(num_partitions)

        return self

    def define_partitioner(self, num_partitions):
        """Define the hash partitioner with the chosen number of partitions for this object's DataFrame.

        Parameters
        ----------
        num_partitions : int
            The target number of partitions of the DataFrame.

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
        """Persist this object's DataFrame considering the chosen storage level.

        Parameters
        ----------
        storage_level : `StorageLevel`, optional
            The desired storage level when materializing the DataFrame. Default value is `StorageLevel.MEMORY_AND_DISK`.

        Returns
        -------
        `self`
            A reference to this object.

        """
        if self.data is not None:
            if not self.isCached():
                self.data.persist(storage_level)
                if self._logger:
                    self._logger.info("DataFrame {} was persisted".format(self.data.id()))
            else:
                if self._logger:
                    self._logger.info("DataFrame {} has already been persisted".format(self.data.id()))
        else:
            if self._logger:
                self._logger.warning("there is no data to be persisted")

        return self

    def unpersist(self):
        """Unpersist this object's DataFrame.

        Returns
        -------
        `self`
            A reference to this object.

        """
        if self.data is not None:
            if self.isCached():
                self.data.unpersist()
                if self._logger:
                    self._logger.info("DataFrame {} was unpersisted".format(self.data.id()))
            else:
                if self._logger:
                    self._logger.info("DataFrame {} has already been unpersisted".format(self.data.id()))
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
        """Materialize this object's DataFrame considering the chosen storage level.

        This method calls persist and right after counts how many elements there are in the DataFrame to force its
        persistence.

        Parameters
        ----------
        storage_level : `StorageLevel`, optional
            The desired storage level when materializing the DataFrame. Default value is `StorageLevel.MEMORY_AND_DISK`.

        Returns
        -------
        `self`
            A reference to this object.

        """
        self.persist(storage_level)
        self._num_nonzero_elements = self.data.count()

        if self._logger:
            self._logger.info("DataFrame {} was materialized".format(self.data.id()))

        return self

    def checkpoint(self):
        """Checkpoint this object's DataFrame.

        Notes
        -----
        If it is intended to use this method in an application, it is necessary to define
        the checkpoint dir using the `SparkSession` object.

        Returns
        -------
        `self`
            A reference to this object.

        """
        if self.isCheckpointed():
            if self._logger:
                self._logger.info("DataFrame already checkpointed")
            return self

        if not self.isCached():
            if self._logger:
                self._logger.warning("it is recommended to cache the DataFrame before checkpointing it")

        self.data.checkpoint()

        if self._logger:
            self._logger.info("DataFrame {} was checkpointed in {}".format(self.data.id(), self.data.getCheckpointFile()))

        self._is_checkpointed = True

        return self

    def dump(self, path, glue=None, codec=None, filename=None):
        """Dump this object's DataFrame to disk in many part-* files.

        Notes
        -----
        Depending, on the chosen dumping mode, this method calls the `collect` method of DataFrame.
        This is not suitable for large working sets, as all data may not fit into main memory.

        Parameters
        ----------
        path : str
            The path where the dumped DataFrame will be located at.
        glue : str, optional
            The glue string that connects each coordinate and value of each element in the DataFrame.
            Default value is `None`. In this case, it uses the 'quantum.dumpingGlue' configuration value.
        codec : str, optional
            Codec name used to compress the dumped data.
            Default value is `None`. In this case, it uses the 'quantum.dumpingCompressionCodec' configuration value.
        filename : str, optional
            File name used when the dumping mode is in a single file. Default value is None.
            In this case, a temporary named file is generated inside the informed path.

        """
        if glue is None:
            glue = Utils.get_conf(self._spark_session, 'quantum.dumpingGlue')

        if codec is None:
            codec = Utils.get_conf(self._spark_session, 'quantum.dumpingCompressionCodec')

        dumping_mode = Utils.get_conf(self._spark_session, 'quantum.math.dumpingMode')

        if dumping_mode == Utils.DumpingModeUniqueFile:
            data = self.data.collect()

            if not filename:
                filename = Utils.get_temp_path(path)

            if len(data):
                with open(Utils.append_slash_dir(path) + filename, 'a') as f:
                    for d in data:
                        f.write(glue.join([str(d[f]) for field in self._fields]))
        elif dumping_mode == Utils.DumpingModePartFiles:
            self.data.map(
                lambda m: glue.join([str(e) for e in m])
            ).saveAsTextFile(path, codec)
        else:
            if self._logger:
                self._logger.error("invalid dumping mode")
            raise NotImplementedError("invalid dumping mode")

    def numpy_array(self):
        """Create a numpy array containing this object's DataFrame data.

        Notes
        -----
        This method calls the `collect` method of DataFrame. This is not suitable for large working sets,
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
