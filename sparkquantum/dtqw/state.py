import math
from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum.math.vector import Vector
from sparkquantum.math.statistics.pdf import is_pdf
from sparkquantum.math.statistics.joint_pdf import JointPDF
from sparkquantum.math.statistics.collision_pdf import CollisionPDF
from sparkquantum.math.statistics.marginal_pdf import MarginalPDF
from sparkquantum.dtqw.coin.coin import is_coin
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.utils.utils import Utils

__all__ = ['State', 'is_state']


class State(Vector):
    """Class for the system state."""

    def __init__(self, rdd, shape, mesh, num_particles):
        """Build a state object.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this state object. Must be a two-dimensional tuple.
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles are walking on.
        num_particles : int
            The number of particles present in the walk.

        """
        if not is_mesh(mesh):
            # self._logger.error("'Mesh' instance expected, not
            # '{}'".format(type(mesh)))
            raise TypeError(
                "'Mesh' instance expected, not '{}'".format(type(mesh)))

        super().__init__(rdd, shape, data_type=complex)

        self._mesh = mesh
        self._num_particles = num_particles

    @property
    def mesh(self):
        """:py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`"""
        return self._mesh

    @property
    def num_particles(self):
        """int"""
        return self._num_particles

    def __str__(self):
        if self._num_particles == 1:
            particles = 'one particle'
        else:
            particles = '{} particles'.format(self._num_particles)

        return 'Quantum State with shape {} of {} over a {}'.format(
            self._shape, particles, self._mesh.to_string())

    def dump(self, path, glue=None, codec=None,
             filename='', dumping_format=None):
        """Dump this object's RDD to disk in a unique file or in many part-* files.

        Notes
        -----
        This method checks the dumping format by using the 'quantum.dtqw.state.dumpingFormat' configuration value.
        In case the chosen format is the mesh coordinates one, this method also checks the state's representation format.
        Depending on the chosen dumping mode, this method calls the :py:func:`pyspark.RDD.collect` method.
        This is not suitable for large working sets, as all data may not fit into driver's main memory.

        Parameters
        ----------
        path : str
            The path where the dumped RDD will be located at.
        glue : str, optional
            The glue string that connects each coordinate and value of each element in the RDD.
            Default value is None. In this case, it uses the 'quantum.dumpingGlue' configuration value.
        codec : str, optional
            Codec name used to compress the dumped data.
            Default value is None. In this case, it uses the 'quantum.dumpingCompressionCodec' configuration value.
        filename : str, optional
            File name used when the dumping mode is in a single file. Default value is None.
            In this case, a temporary named file is generated inside the informed path.
        dumping_format : int, optional
            Printing format used to dump this state.
            Default value is None. In this case, it uses the 'quantum.math.dumpingFormat' configuration value.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If any of the chosen 'quantum.dtqw.state.dumpingFormat', 'quantum.math.dumpingMode' or
            'quantum.dtqw.state.representationFormat' configuration is not valid.

        """
        if glue is None:
            glue = Utils.get_conf(self._spark_context, 'quantum.dumpingGlue')

        if codec is None:
            codec = Utils.get_conf(self._spark_context,
                                   'quantum.dumpingCompressionCodec')

        if dumping_format is None:
            dumping_format = int(Utils.get_conf(
                self._spark_context, 'quantum.dtqw.state.dumpingFormat'))

        dumping_mode = int(Utils.get_conf(
            self._spark_context, 'quantum.math.dumpingMode'))

        if dumping_format == Utils.StateDumpingFormatIndex:
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
                if self._logger is not None:
                    self._logger.error("invalid dumping mode")
                raise ValueError("invalid dumping mode")
        elif dumping_format == Utils.StateDumpingFormatCoordinate:
            repr_format = int(Utils.get_conf(
                self._spark_context, 'quantum.dtqw.state.representationFormat'))

            if self._mesh.is_1d():
                ndim = self._mesh.dimension
                coin_size = self._mesh.coin_size
                size = self._mesh.size
                num_particles = self._num_particles
                size_per_coin = int(coin_size / ndim)
                cs_size = size_per_coin * size

                mesh_offset = min(self._mesh.axis())

                if repr_format == Utils.StateRepresentationFormatCoinPosition:
                    def __map(m):
                        ix = []

                        for p in range(num_particles):
                            # Coin
                            ix.append(
                                str(int(m[0] / (cs_size ** (num_particles - 1 - p) * size)) % size_per_coin))
                            # Position
                            ix.append(
                                str(int(m[0] / (cs_size ** (num_particles - 1 - p))) % size + mesh_offset))

                        ix.append(str(m[1]))

                        return glue.join(ix)
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                    def __map(m):
                        xi = []

                        for p in range(num_particles):
                            # Position
                            xi.append(str(int(
                                m[0] / (cs_size ** (num_particles - 1 - p) * size_per_coin)) % size + mesh_offset))
                            # Coin
                            xi.append(
                                str(int(m[0] / (cs_size ** (num_particles - 1 - p))) % size_per_coin))

                        xi.append(str(m[1]))

                        return glue.join(xi)
                else:
                    if self._logger is not None:
                        self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")
            elif self._mesh.is_2d():
                ndim = self._mesh.dimension
                coin_size = self._mesh.coin_size
                size_x, size_y = self._mesh.size
                num_particles = self._num_particles
                size_per_coin = int(coin_size / ndim)
                cs_size_x = size_per_coin * size_x
                cs_size_y = size_per_coin * size_y
                cs_size_xy = cs_size_x * cs_size_y

                axis = self._mesh.axis()
                mesh_offset_x, mesh_offset_y = axis[0][0][0], axis[1][0][0]

                if repr_format == Utils.StateRepresentationFormatCoinPosition:
                    def __map(m):
                        ijxy = []

                        for p in range(num_particles):
                            # Coin
                            ijxy.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * cs_size_x * size_y)) % size_per_coin))
                            ijxy.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * size_x * size_y)) % size_per_coin))
                            # Position
                            ijxy.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * size_y)) % size_x + mesh_offset_x))
                            ijxy.append(
                                str(int(m[0] / (cs_size_xy ** (num_particles - 1 - p))) % size_y + mesh_offset_y))

                        ijxy.append(str(m[1]))

                        return glue.join(ijxy)
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                    def __map(m):
                        xyij = []

                        for p in range(num_particles):
                            # Position
                            xyij.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * coin_size * size_y)) % size_x + mesh_offset_x))
                            xyij.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * coin_size)) % size_y + mesh_offset_y))
                            # Coin
                            xyij.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * size_per_coin)) % size_per_coin))
                            xyij.append(
                                str(int(m[0] / (cs_size_xy ** (num_particles - 1 - p))) % size_per_coin))

                        xyij.append(str(m[1]))

                        return glue.join(xyij)
                else:
                    if self._logger is not None:
                        self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")
            else:
                if self._logger is not None:
                    self._logger.error("mesh dimension not implemented")
                raise NotImplementedError("mesh dimension not implemented")

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
                            f.write(glue.join(d) + "\n")
            elif dumping_mode == Utils.DumpingModePartFiles:
                self.data.map(
                    __map
                ).saveAsTextFile(path, codec)
            else:
                if self._logger is not None:
                    self._logger.error("invalid dumping mode")
                raise ValueError("invalid dumping mode")
        else:
            if self._logger is not None:
                self._logger.error("invalid dumping format")
            raise ValueError("invalid dumping format")

    def kron(self, other):
        """Perform a tensor (Kronecker) product with another system state.

        Parameters
        ----------
        other : :py:class:`sparkquantum.dtqw.state.State`
            The other system state.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.state.State`
            The resulting state.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.dtqw.state.State`.

        """
        if not is_state(other):
            if self._logger is not None:
                self._logger.error(
                    "'State' instance expected, not '{}'".format(type(other)))
            raise TypeError(
                "'State' instance expected, not '{}'".format(type(other)))

        rdd, new_shape = self._kron(other)

        return State(rdd, new_shape, self._mesh, self._num_particles)

    def measure_system(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the entire system state.

        Parameters
        ----------
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.math.statistics.JointPDF`
            The PDF of the entire system.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid or if the sum of the calculated PDF is not equal to one.

        """
        if self._logger is not None:
            self._logger.info("measuring the state of the system...")

        t1 = datetime.now()

        repr_format = int(Utils.get_conf(self._spark_context,
                                         'quantum.dtqw.state.representationFormat'))

        if self._mesh.is_1d():
            ndim = self._mesh.dimension
            coin_size = self._mesh.coin_size
            size = self._mesh.size
            num_particles = self._num_particles
            ind = ndim * num_particles
            expected_elems = size
            size_per_coin = int(coin_size / ndim)
            cs_size = size_per_coin * size
            dims = [size for p in range(ind)]

            if self._num_particles == 1:
                dims.append(1)

            shape = tuple(dims)

            if repr_format == Utils.StateRepresentationFormatCoinPosition:
                def __map(m):
                    x = []

                    for p in range(num_particles):
                        x.append(
                            int(m[0] / (cs_size ** (num_particles - 1 - p))) % size)

                    return tuple(x), (abs(m[1]) ** 2).real
            elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                def __map(m):
                    x = []

                    for p in range(num_particles):
                        x.append(
                            int(m[0] / (cs_size ** (num_particles - 1 - p) * size_per_coin)) % size)

                    return tuple(x), (abs(m[1]) ** 2).real
            else:
                if self._logger is not None:
                    self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                a = []

                for p in range(num_particles):
                    a.append(m[0][p])

                a.append(m[1])

                return tuple(a)
        elif self._mesh.is_2d():
            ndim = self._mesh.dimension
            coin_size = self._mesh.coin_size
            size_x, size_y = self._mesh.size
            num_particles = self._num_particles
            ind = ndim * num_particles
            expected_elems = size_x * size_y
            size_per_coin = int(coin_size / ndim)
            cs_size_x = size_per_coin * size_x
            cs_size_y = size_per_coin * size_y
            cs_size_xy = cs_size_x * cs_size_y
            dims = []

            for p in range(0, ind, ndim):
                dims.append(self._mesh.size[0])
                dims.append(self._mesh.size[1])

            shape = tuple(dims)

            if repr_format == Utils.StateRepresentationFormatCoinPosition:
                def __map(m):
                    xy = []

                    for p in range(num_particles):
                        xy.append(
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p) * size_y)) % size_x)
                        xy.append(
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p))) % size_y)

                    return tuple(xy), (abs(m[1]) ** 2).real
            elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                def __map(m):
                    xy = []

                    for p in range(num_particles):
                        xy.append(
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p) * coin_size * size_y)) % size_x)
                        xy.append(
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p) * coin_size)) % size_y)

                    return tuple(xy), (abs(m[1]) ** 2).real
            else:
                if self._logger is not None:
                    self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                xy = []

                for p in range(0, ind, ndim):
                    xy.append(m[0][p])
                    xy.append(m[0][p + 1])

                xy.append(m[1])

                return tuple(xy)
        else:
            if self._logger is not None:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        expected_size = Utils.get_size_of_type(float) * expected_elems
        num_partitions = Utils.get_num_partitions(
            self.data.context, expected_size)

        data_type = self._data_type()

        rdd = self.data.filter(
            lambda m: m[1] != data_type
        ).map(
            __map
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        ).map(
            __unmap
        )

        pdf = JointPDF(rdd, shape, self._mesh,
                       self._num_particles).materialize(storage_level)

        if self._logger is not None:
            self._logger.info("checking if the probabilities sum one...")

        round_precision = int(Utils.get_conf(
            self._spark_context, 'quantum.math.roundPrecision'))

        if round(pdf.sum_values(), round_precision) != 1.0:
            if self._logger is not None:
                self._logger.error("PDFs must sum one")
            raise ValueError("PDFs must sum one")

        app_id = self._spark_context.applicationId

        if self._profiler is not None:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_pdf(
                'fullMeasurement', pdf, (datetime.now() - t1).total_seconds())

            if self._logger is not None:
                self._logger.info(
                    "full measurement was done in {}s".format(info['buildingTime']))
                self._logger.info(
                    "PDF with full measurement is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

            self._profiler.log_rdd(app_id=app_id)

        return pdf

    def measure_collision(self, full_measurement,
                          storage_level=StorageLevel.MEMORY_AND_DISK):
        """Filter the measurement of the entire system by checking when
        all particles are located at the same site of the mesh.

        Parameters
        ----------
        full_measurement : :py:class:`sparkquantum.math.statistics.pdf.PDF`
            The measurement of the entire system.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.math.statistics.collision_pdf.CollisionPDF`
            The PDF of the system when all particles are located at the same site.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid or if the system is composed by only one particle.

        TypeError
            If `full_measurement` is not valid.

        """
        if self._num_particles <= 1:
            if self._logger is not None:
                self._logger.error(
                    "the measurement of collision cannot be performed for quantum walks with only one particle")
            raise NotImplementedError(
                "the measurement of collision cannot be performed for quantum walks with only one particle")

        if self._logger is not None:
            self._logger.info(
                "measuring the state of the system considering that the particles are at the same positions...")

        t1 = datetime.now()

        if not is_pdf(full_measurement):
            if self._logger is not None:
                self._logger.error(
                    "'PDF' instance expected, not '{}'".format(type(full_measurement)))
            raise TypeError("'PDF' instance expected, not '{}'".format(
                type(full_measurement)))

        if self._mesh.is_1d():
            ndim = self._mesh.dimension
            size = self._mesh.size
            num_particles = self._num_particles
            ind = ndim * num_particles
            expected_elems = size
            shape = (size, 1)

            def __filter(m):
                for p in range(num_particles):
                    if m[0] != m[p]:
                        return False
                return True

            def __map(m):
                return m[0], m[ind]
        elif self._mesh.is_2d():
            ndim = self._mesh.dimension
            size_x, size_y = self._mesh.size
            num_particles = self._num_particles
            ind = ndim * num_particles
            expected_elems = size_x * size_y
            shape = (size_x, size_y)

            def __filter(m):
                for p in range(0, ind, ndim):
                    if m[0] != m[p] or m[1] != m[p + 1]:
                        return False
                return True

            def __map(m):
                return m[0], m[1], m[ind]
        else:
            if self._logger is not None:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        expected_size = Utils.get_size_of_type(float) * expected_elems
        num_partitions = Utils.get_num_partitions(
            self.data.context, expected_size)

        rdd = full_measurement.data.filter(
            __filter
        ).map(
            __map
        ).coalesce(
            num_partitions
        )

        pdf = CollisionPDF(rdd, shape, self._mesh,
                           self._num_particles).materialize(storage_level)

        app_id = self._spark_context.applicationId

        if self._profiler is not None:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_pdf(
                'collisionMeasurement', pdf, (datetime.now() - t1).total_seconds())

            if self._logger is not None:
                self._logger.info(
                    "collision measurement was done in {}s".format(info['buildingTime']))
                self._logger.info(
                    "PDF with collision measurement is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

            self._profiler.log_rdd(app_id=app_id)

        return pdf

    def measure_particle(self, particle,
                         storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the partial measurement of a particle of the system state.

        Parameters
        ----------
        particle : int
            The desired particle to be measured. The particle number starts by 0.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.math.statistics.marginal_pdf.MarginalPDF`
            The PDF of each particle.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If `particle` is not valid, i.e., particle number does not belong to the walk,
            if the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid or
            if the sum of the calculated PDF is not equal to one.

        """
        if particle < 0 or particle >= self._num_particles:
            if self._logger is not None:
                self._logger.error("invalid particle number")
            raise ValueError("invalid particle number")

        if self._logger is not None:
            self._logger.info(
                "measuring the state of the system for particle {}...".format(particle + 1))

        t1 = datetime.now()

        repr_format = int(Utils.get_conf(self._spark_context,
                                         'quantum.dtqw.state.representationFormat'))

        if self._mesh.is_1d():
            ndim = self._mesh.dimension
            coin_size = self._mesh.coin_size
            size = self._mesh.size
            num_particles = self._num_particles
            expected_elems = size
            size_per_coin = int(coin_size / ndim)
            cs_size = size_per_coin * size
            shape = (size, 1)

            if repr_format == Utils.StateRepresentationFormatCoinPosition:
                def __map(m):
                    x = int(
                        m[0] / (cs_size ** (num_particles - 1 - particle))) % size
                    return x, (abs(m[1]) ** 2).real
            elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                def __map(m):
                    x = int(m[0] / (cs_size ** (num_particles -
                                                1 - particle) * size_per_coin)) % size
                    return x, (abs(m[1]) ** 2).real
            else:
                if self._logger is not None:
                    self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                return m
        elif self._mesh.is_2d():
            ndim = self._mesh.dimension
            coin_size = self._mesh.coin_size
            size_x, size_y = self._mesh.size
            num_particles = self._num_particles
            expected_elems = size_x * size_y
            size_per_coin = int(coin_size / ndim)
            cs_size_x = size_per_coin * size_x
            cs_size_y = size_per_coin * size_y
            cs_size_xy = cs_size_x * cs_size_y
            shape = (size_x, size_y)

            if repr_format == Utils.StateRepresentationFormatCoinPosition:
                def __map(m):
                    xy = (
                        int(m[0] / (cs_size_xy ** (num_particles -
                                                   1 - particle) * size_y)) % size_x,
                        int(m[0] / (cs_size_xy **
                                    (num_particles - 1 - particle))) % size_y
                    )
                    return xy, (abs(m[1]) ** 2).real
            elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                def __map(m):
                    xy = (
                        int(m[0] / (cs_size_xy ** (num_particles - 1 -
                                                   particle) * coin_size * size_y)) % size_x,
                        int(m[0] / (cs_size_xy ** (num_particles -
                                                   1 - particle) * coin_size)) % size_y
                    )
                    return xy, (abs(m[1]) ** 2).real
            else:
                if self._logger is not None:
                    self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                return m[0][0], m[0][1], m[1]
        else:
            if self._logger is not None:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        expected_size = Utils.get_size_of_type(float) * expected_elems
        num_partitions = Utils.get_num_partitions(
            self.data.context, expected_size)

        data_type = self._data_type()

        rdd = self.data.filter(
            lambda m: m[1] != data_type
        ).map(
            __map
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        ).map(
            __unmap
        )

        pdf = MarginalPDF(rdd, shape, self._mesh,
                          self._num_particles).materialize(storage_level)

        if self._logger is not None:
            self._logger.info("checking if the probabilities sum one...")

        round_precision = int(Utils.get_conf(
            self._spark_context, 'quantum.math.roundPrecision'))

        if round(pdf.sum_values(), round_precision) != 1.0:
            if self._logger is not None:
                self._logger.error("PDFs must sum one")
            raise ValueError("PDFs must sum one")

        app_id = self._spark_context.applicationId

        if self._profiler is not None:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_pdf(
                'partialMeasurementParticle{}'.format(
                    particle + 1), pdf, (datetime.now() - t1).total_seconds()
            )

            if self._logger is not None:
                self._logger.info("partial measurement for particle {} was done in {}s".format(
                    particle + 1, info['buildingTime'])
                )
                self._logger.info(
                    "PDF with partial measurements for particle {} "
                    "are consuming {} bytes in memory and {} bytes in disk".format(
                        particle + 1, info['memoryUsed'], info['diskUsed']
                    )
                )

            self._profiler.log_rdd(app_id=app_id)

        return pdf

    def measure_particles(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the partial measurement of each particle of the system state.

        Parameters
        ----------
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        tuple
            A tuple containing the PDF of each particle.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If `particle` is not valid, i.e., particle number does not belong to the walk,
            if the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid or
            if the sum of the calculated PDF is not equal to one.

        """
        return [self.measure_particle(p, storage_level)
                for p in range(self._num_particles)]

    def measure(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the system state.

        If the state is composed by only one particle, the full measurement of the
        system is performed and returned. In other cases, the measurement process will return a tuple containing
        the full measurement, the collision measurement - probabilities of each mesh site
        with all particles located at - and the partial measurement of each particle.

        Parameters
        ----------
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.math.statistics.pdf.PDF` or tuple
            :py:class:`sparkquantum.math.statistics.pdf.PDF` if the system is composed by only one particle, tuple otherwise.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid or
            if the sum of the calculated PDF is not equal to one.

        """
        if self._num_particles == 1:
            return self.measure_system(storage_level)
        else:
            full_measurement = self.measure_system(storage_level)
            collision_measurement = self.measure_collision(
                full_measurement, storage_level)
            partial_measurements = self.measure_particles(storage_level)

            return full_measurement, collision_measurement, partial_measurements

    @staticmethod
    def create(coin, mesh, positions, amplitudes,
               representationFormat=Utils.StateRepresentationFormatCoinPosition, logger=None):
        """Create a system state.

        For system states with entangled particles, the state must be created
        by its class construct method.

        Parameters
        ----------
        coin : :py:class:`sparkquantum.dtqw.coin.coin.Coin`
            A coin object.
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particle(s) is(are) walking on.
        positions : tuple or list
            The position of each particle present in the quantum walk.
        amplitudes : tuple or list
            The amplitudes for each qubit of each particle in the quantum walk.
        representationFormat : int, optional
            Indicate how the quantum system will be represented.
            Default value is :py:const:`sparkquantum.utils.Utils.StateRepresentationFormatCoinPosition`.
        logger : py:class:`sparkquantum.utils.logger.Logger`, optional
            A logger object

        Returns
        -------
        :py:class:`sparkquantum.dtqw.state.State`
            A new system state.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid.

        TypeError
            If `coin` is not a :py:class:`sparkquantum.dtqw.coin.coin.Coin` or
            if `mesh` is not a :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`.

        """
        if not is_coin(coin):
            if logger is not None:
                logger.error(
                    "'Coin' instance expected, not '{}'".format(type(coin)))
            raise TypeError(
                "'Coin' instance expected, not '{}'".format(type(coin)))

        if not is_mesh(mesh):
            if logger is not None:
                logger.error(
                    "'Mesh' instance expected, not '{}'".format(type(mesh)))
            raise TypeError(
                "'Mesh' instance expected, not '{}'".format(type(mesh)))

        spark_context = SparkContext.getOrCreate()

        coin_size = coin.size

        if mesh.is_1d():
            mesh_size = mesh.size
        elif mesh.is_2d():
            mesh_size = mesh.size[0] * mesh.size[1]
        else:
            if logger is not None:
                logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        base_states = []

        num_particles = len(positions)

        shape = (coin_size * mesh_size, 1)

        for p in range(num_particles):
            if representationFormat == Utils.StateRepresentationFormatCoinPosition:
                state = (
                    (a * mesh_size + positions[p],
                     amplitudes[p][a]) for a in range(
                        len(amplitudes[p])))
            elif representationFormat == Utils.StateRepresentationFormatPositionCoin:
                state = (
                    (positions[p] * coin_size + a,
                     amplitudes[p][a]) for a in range(
                        len(amplitudes[p])))
            else:
                if logger is not None:
                    logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            rdd = spark_context.parallelize(state)

            base_states.append(
                State(
                    rdd,
                    shape,
                    mesh,
                    num_particles))

        initial_state = base_states[0]

        for p in range(1, num_particles):
            initial_state = initial_state.kron(base_states[p])

        for bs in base_states:
            bs.destroy()

        return initial_state


def is_state(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.state.State` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.state.State` object, False otherwise.

    """
    return isinstance(obj, State)
