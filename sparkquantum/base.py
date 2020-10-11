from pyspark import RDD, StorageLevel

from sparkquantum import util

__all__ = ['Base']


class Base:
    """Top-level class to act as a container of :py:class:`pyspark.RDD`."""

    def __init__(self, rdd, nelem=None):
        """Build a top-level object that acts as a container of :py:class:`pyspark.RDD`.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        nelem : int, optional
            The expected (or definitive) number of elements. This helps to find a
            better number of partitions when (re)partitioning the RDD. Default value is None.

        """
        self._sc = rdd.context
        self._data = rdd
        self._nelem = nelem

        self._logger = util.get_logger(
            self._sc, self.__class__.__name__)

        if not isinstance(rdd, RDD):
            self._logger.error(
                "'RDD' instance expected, not '{}'".format(
                    type(rdd)))
            raise TypeError(
                "'RDD' instance expected, not '{}'".format(
                    type(rdd)))

    @property
    def sc(self):
        """:py:class:`pyspark.SparkContext`"""
        return self._sc

    @property
    def nelem(self):
        """int"""
        return self._nelem

    @property
    def data(self):
        """:py:class:`pyspark.RDD`"""
        return self._data

    def __del__(self):
        # In cases where multiple simulations are performed,
        # the same Python's logger object is used for the same class name.
        # Removing all its handlers, the object is reset.
        for h in self._logger.handlers:
            self._logger.removeHandler(h)

    def __str__(self):
        """Build a string representing this math base entity.

        Returns
        -------
        str
            The string representation of this math base entity.

        """
        return 'Base math entity'

    def repartition(self, num_partitions, shuffle=False):
        """Change the number of partitions of this object's RDD.

        Parameters
        ----------
        num_partitions : int
            The target number of partitions of the RDD.
        shuffle: bool
            Indicate that Spark must force a shuffle operation.

        Returns
        -------
        :py:class:`sparkquantum.base.Base`
            A reference to this object.

        """
        if num_partitions > self._data.getNumPartitions():
            self._data = self._data.repartition(num_partitions)
        elif num_partitions < self._data.getNumPartitions():
            self._data = self._data.coalesce(num_partitions, shuffle)

        return self

    def partition_by(self, num_partitions=None, partition_func=None):
        """Set a partitioner with the chosen number of partitions for this object's RDD.

        Notes
        -----
        When `partition_func` is None, the default partition function is used (i.e., portable_hash).

        Parameters
        ----------
        num_partitions : int, optional
            The chosen number of partitions for the RDD.
            Default value is the original number of partitions of the RDD.
        partition_func: function, optional
            The chosen partition function.
            Default value is None.

        Returns
        -------
        :py:class:`sparkquantum.base.Base`
            A reference to this object.

        """
        if num_partitions is None:
            np = self._data.getNumPartitions()
        else:
            np = num_partitions

        if partition_func is None:
            self._data = self._data.partitionBy(np)
        else:
            self._data = self._data.partitionBy(
                np, partitionFunc=partition_func
            )

        return self

    def persist(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Persist this object's RDD considering the chosen storage level.

        Parameters
        ----------
        storage_level : :py:class:`pyspark.StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is :py:const:`pyspark.StorageLevel.MEMORY_AND_DISK`.

        Returns
        -------
        :py:class:`sparkquantum.base.Base`
            A reference to this object.

        """
        if self._data is not None:
            if not self._data.is_cached:
                self._data.persist(storage_level)
                self._logger.info(
                    "RDD {} was persisted".format(
                        self._data.id()))
            else:
                self._logger.info(
                    "RDD {} has already been persisted".format(
                        self._data.id()))
        else:
            self._logger.warning(
                "there is no data to be persisted")

        return self

    def unpersist(self):
        """Unpersist this object's RDD.

        Returns
        -------
        :py:class:`sparkquantum.base.Base`
            A reference to this object.

        """
        if self._data is not None:
            if self._data.is_cached:
                self._data.unpersist()
                self._logger.info(
                    "RDD {} was unpersisted".format(
                        self._data.id()))
            else:
                self._logger.info(
                    "RDD {} has already been unpersisted".format(
                        self._data.id()))
        else:
            self._logger.warning(
                "there is no data to be unpersisted")

        return self

    def destroy(self):
        """Alias of :py:func:`unpersist`.

        Returns
        -------
        :py:class:`sparkquantum.base.Base`
            A reference to this object.

        """
        return self.unpersist()

    def materialize(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Materialize this object's RDD considering the chosen storage level.

        This method calls persist and right after counts how many elements there are in the RDD to force its
        persistence.

        Parameters
        ----------
        storage_level : :py:class:`pyspark.StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is :py:const:`pyspark.StorageLevel.MEMORY_AND_DISK`.

        Returns
        -------
        :py:class:`sparkquantum.base.Base`
            A reference to this object.

        """
        self.persist(storage_level=storage_level)

        self._nelem = self._data.count()

        self._logger.info("RDD {} was materialized".format(self._data.id()))

        return self

    def checkpoint(self):
        """Checkpoint this object's RDD.

        Notes
        -----
        If it is intended to use this method in an application, it is necessary to define
        the checkpoint dir using the :py:class:`pyspark.SparkContext` object.

        Returns
        -------
        :py:class:`sparkquantum.base.Base`
            A reference to this object.

        """
        if self._data.isCheckpointed():
            self._logger.info("RDD already checkpointed")
            return self

        if not self._data.is_cached:
            self._logger.warning(
                "it is recommended to cache the RDD before checkpointing it")

        self._data.checkpoint()

        self._logger.info(
            "RDD {} was checkpointed in {}".format(
                self._data.id(),
                self._data.getCheckpointFile()))

        return self
