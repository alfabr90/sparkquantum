import math
from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.math.vector import Vector
from sparkquantum.math.statistics.pdf import is_pdf
from sparkquantum.math.statistics.joint_pdf import JointPDF
from sparkquantum.math.statistics.collision_pdf import CollisionPDF
from sparkquantum.math.statistics.marginal_pdf import MarginalPDF
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.utils.utils import Utils

__all__ = ['State', 'is_state']


class State(Vector):
    """Class for the system state."""

    def __init__(self, rdd, shape, mesh, num_particles):
        """Build a `State` object.

        Parameters
        ----------
        rdd : `RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this state object. Must be a 2-dimensional tuple.
        mesh : `Mesh`
            The mesh where the particles are walking on.
        num_particles : int
            The number of particles present in the walk.

        """
        if not is_mesh(mesh):
            # self._logger.error("'Mesh' instance expected, not '{}'".format(type(mesh)))
            raise TypeError("'Mesh' instance expected, not '{}'".format(type(mesh)))

        super().__init__(rdd, shape, data_type=complex)

        self._mesh = mesh
        self._num_particles = num_particles

    @property
    def mesh(self):
        """`Mesh`"""
        return self._mesh

    @property
    def num_particles(self):
        """int"""
        return self._num_particles

    def kron(self, other):
        """Perform a tensor (Kronecker) product with another system state.

        Parameters
        ----------
        other : `State`
            The other system state.

        Returns
        -------
        `State`
            The resulting state.

        """
        if not is_state(other):
            if self._logger:
                self._logger.error("`State` instance expected, not '{}'".format(type(other)))
            raise TypeError("`State` instance expected, not '{}'".format(type(other)))

        rdd, new_shape = self._kron(other)

        return State(rdd, new_shape, self._mesh, self._num_particles)

    def measure_system(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the entire system state.

        Parameters
        ----------
        storage_level : `StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        `JointPDF`
            The PDF of the entire system.

        Raises
        ------
        `NotImplementedError`
        `ValueError`

        """
        if self._logger:
            self._logger.info("measuring the state of the system...")

        t1 = datetime.now()

        repr_format = int(Utils.get_conf(self._spark_context, 'quantum.representationFormat', default=Utils.RepresentationFormatCoinPosition))

        if self._mesh.is_1d():
            ndim = self._mesh.dimension
            coin_size = self._mesh.coin_size
            size = self._mesh.size
            num_particles = self._num_particles
            ind = ndim * num_particles
            expected_elems = size
            cs_size = int(coin_size / ndim) * size
            dims = [size for p in range(ind)]

            if self._num_particles == 1:
                dims.append(1)

            shape = tuple(dims)

            if repr_format == Utils.RepresentationFormatCoinPosition:
                def __map(m):
                    x = []

                    for p in range(num_particles):
                        x.append(int(m[0] / (cs_size ** (num_particles - 1 - p))) % size)

                    return tuple(x), (abs(m[1]) ** 2).real
            elif repr_format == Utils.RepresentationFormatPositionCoin:
                def __map(m):
                    x = []

                    for p in range(num_particles):
                        x.append(int(m[0] / (cs_size ** (num_particles - 1 - p) * coin_size)) % size)

                    return tuple(x), (abs(m[1]) ** 2).real
            else:
                if self._logger:
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

            if repr_format == Utils.RepresentationFormatCoinPosition:
                def __map(m):
                    xy = []

                    for p in range(num_particles):
                        xy.append(int(m[0] / (cs_size_xy ** (num_particles - 1 - p) * size_y)) % size_x)
                        xy.append(int(m[0] / (cs_size_xy ** (num_particles - 1 - p))) % size_y)

                    return tuple(xy), (abs(m[1]) ** 2).real
            elif repr_format == Utils.RepresentationFormatPositionCoin:
                def __map(m):
                    xy = []

                    for p in range(num_particles):
                        xy.append(int(m[0] / (cs_size_xy ** (num_particles - 1 - p) * coin_size * size_y)) % size_x)
                        xy.append(int(m[0] / (cs_size_xy ** (num_particles - 1 - p) * coin_size)) % size_y)

                    return tuple(xy), (abs(m[1]) ** 2).real
            else:
                if self._logger:
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
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        expected_size = Utils.get_size_of_type(float) * expected_elems
        num_partitions = Utils.get_num_partitions(self.data.context, expected_size)

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

        pdf = JointPDF(rdd, shape, self._mesh, self._num_particles).materialize(storage_level)

        if self._logger:
            self._logger.info("checking if the probabilities sum one...")

        round_precision = int(Utils.get_conf(self._spark_context, 'quantum.math.roundPrecision', default='10'))

        if round(pdf.sum_values(), round_precision) != 1.0:
            if self._logger:
                self._logger.error("PDFs must sum one")
            raise ValueError("PDFs must sum one")

        app_id = self._spark_context.applicationId

        if self._profiler:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_pdf('fullMeasurement', pdf, (datetime.now() - t1).total_seconds())

            if self._logger:
                self._logger.info("full measurement was done in {}s".format(info['buildingTime']))
                self._logger.info(
                    "PDF with full measurement is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

            self._profiler.log_rdd(app_id=app_id)

        return pdf

    def measure_collision(self, full_measurement, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Filter the measurement of the entire system by checking when
        all particles are located at the same site of the mesh.

        Parameters
        ----------
        full_measurement : `PDF`
            The measurement of the entire system.
        storage_level : `StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        `CollisionPDF`
            The PDF of the system when all particles are located at the same site.

        Raises
        ------
        `NotImplementedError`

        """
        if self._num_particles <= 1:
            if self._logger:
                self._logger.error("the measurement of collision cannot be performed for quantum walks with only one particle")
            raise NotImplementedError("the measurement of collision cannot be performed for quantum walks with only one particle")

        if self._logger:
            self._logger.info("measuring the state of the system considering that the particles are at the same positions...")

        t1 = datetime.now()

        if not is_pdf(full_measurement):
            if self._logger:
                self._logger.error("'PDF' instance expected, not '{}'".format(type(full_measurement)))
            raise TypeError("'PDF' instance expected, not '{}'".format(type(full_measurement)))

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
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        expected_size = Utils.get_size_of_type(float) * expected_elems
        num_partitions = Utils.get_num_partitions(self.data.context, expected_size)

        rdd = full_measurement.data.filter(
            __filter
        ).map(
            __map
        ).coalesce(
            num_partitions
        )

        pdf = CollisionPDF(rdd, shape, self._mesh, self._num_particles).materialize(storage_level)

        app_id = self._spark_context.applicationId

        if self._profiler:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_pdf('collisionMeasurement', pdf, (datetime.now() - t1).total_seconds())

            if self._logger:
                self._logger.info("collision measurement was done in {}s".format(info['buildingTime']))
                self._logger.info(
                    "PDF with collision measurement is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

            self._profiler.log_rdd(app_id=app_id)

        return pdf

    def measure_particle(self, particle, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the partial measurement of a particle of the system state.

        Parameters
        ----------
        particle : int
            The desired particle to be measured. The particle number starts by 0.
        storage_level : `StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        `MarginalPDF`
            The PDF of each particle.

        Raises
        ------
        `NotImplementedError`
        `ValueError`

        """
        if particle >= self._num_particles:
            if self._logger:
                self._logger.error("invalid particle number")
            raise ValueError("invalid particle number")

        if self._logger:
            self._logger.info("measuring the state of the system for particle {}...".format(particle + 1))

        t1 = datetime.now()

        repr_format = int(Utils.get_conf(self._spark_context, 'quantum.representationFormat', default=Utils.RepresentationFormatCoinPosition))

        if self._mesh.is_1d():
            ndim = self._mesh.dimension
            coin_size = self._mesh.coin_size
            size = self._mesh.size
            num_particles = self._num_particles
            expected_elems = size
            cs_size = int(coin_size / ndim) * size
            shape = (size, 1)

            if repr_format == Utils.RepresentationFormatCoinPosition:
                def __map(m):
                    x = int(m[0] / (cs_size ** (num_particles - 1 - particle))) % size
                    return x, (abs(m[1]) ** 2).real
            elif repr_format == Utils.RepresentationFormatPositionCoin:
                def __map(m):
                    x = int(m[0] / (cs_size ** (num_particles - 1 - particle) * coin_size)) % size
                    return x, (abs(m[1]) ** 2).real
            else:
                if self._logger:
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
            cs_size_x = int(coin_size / ndim) * size_x
            cs_size_y = int(coin_size / ndim) * size_y
            cs_size_xy = cs_size_x * cs_size_y
            shape = (size_x, size_y)

            if repr_format == Utils.RepresentationFormatCoinPosition:
                def __map(m):
                    xy = (
                        int(m[0] / (cs_size_xy ** (num_particles - 1 - particle) * size_y)) % size_x,
                        int(m[0] / (cs_size_xy ** (num_particles - 1 - particle))) % size_y
                    )
                    return xy, (abs(m[1]) ** 2).real
            elif repr_format == Utils.RepresentationFormatPositionCoin:
                def __map(m):
                    xy = (
                        int(m[0] / (cs_size_xy ** (num_particles - 1 - particle) * coin_size * size_y)) % size_x,
                        int(m[0] / (cs_size_xy ** (num_particles - 1 - particle) * coin_size)) % size_y
                    )
                    return xy, (abs(m[1]) ** 2).real
            else:
                if self._logger:
                    self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                return m[0][0], m[0][1], m[1]
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        expected_size = Utils.get_size_of_type(float) * expected_elems
        num_partitions = Utils.get_num_partitions(self.data.context, expected_size)

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

        pdf = MarginalPDF(rdd, shape, self._mesh, self._num_particles).materialize(storage_level)

        if self._logger:
            self._logger.info("checking if the probabilities sum one...")

        round_precision = int(Utils.get_conf(self._spark_context, 'quantum.math.roundPrecision', default='10'))

        if round(pdf.sum_values(), round_precision) != 1.0:
            if self._logger:
                self._logger.error("PDFs must sum one")
            raise ValueError("PDFs must sum one")

        app_id = self._spark_context.applicationId

        if self._profiler:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_pdf(
                'partialMeasurementParticle{}'.format(particle + 1), pdf, (datetime.now() - t1).total_seconds()
            )

            if self._logger:
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
        storage_level : `StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        tuple
            A tuple containing the PDF of each particle.

        """
        return [self.measure_particle(p, storage_level) for p in range(self._num_particles)]

    def measure(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the system state.

        If the state is composed by only one particle, the full measurement of the
        system is performed and returned. In other cases, the measurement process will return a tuple containing
        the full measurement, the collision measurement - probabilities of each mesh site
        with all particles located at - and the partial measurement of each particle.

        Parameters
        ----------
        storage_level : `StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        `PDF` or tuple
            `PDF` if the system is composed by only one particle, tuple otherwise.

        """
        if self._num_particles == 1:
            return self.measure_system(storage_level)
        else:
            full_measurement = self.measure_system(storage_level)
            collision_measurement = self.measure_collision(full_measurement, storage_level)
            partial_measurements = self.measure_particles(storage_level)

            return full_measurement, collision_measurement, partial_measurements


def is_state(obj):
    """Check whether argument is a `State` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a `State` object, False otherwise.

    """
    return isinstance(obj, State)
