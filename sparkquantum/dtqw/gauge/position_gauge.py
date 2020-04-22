from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.dtqw.gauge.gauge import Gauge
from sparkquantum.math.statistics.collision_pdf import CollisionPDF
from sparkquantum.math.statistics.joint_pdf import JointPDF
from sparkquantum.math.statistics.marginal_pdf import MarginalPDF
from sparkquantum.math.statistics.pdf import is_pdf
from sparkquantum.utils.utils import Utils

__all__ = ['PositionGauge']


class PositionGauge(Gauge):
    """Top-level class for system state's positions measurements (gauge)."""

    def __init__(self):
        """Build a top-level system state's positions measurements (gauge) object."""
        super().__init__()

    def measure_system(
            self, state, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the entire system state.

        Parameters
        ----------
        state : :py:class:`sparkquantum.dtqw.state.State`
            A system state.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.math.statistics.JointPDF`
            The PDF with all possible positions of the entire system.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If the chosen 'quantum.dtqw.state.representationFormat' configuration
            is not valid or if the sum of the calculated PDF is not equal to one.

        """
        self._logger.info("measuring the state of the system...")

        t1 = datetime.now()

        repr_format = int(Utils.get_conf(self._spark_context,
                                         'quantum.dtqw.state.representationFormat'))

        if state.mesh.is_1d():
            ndim = state.mesh.dimension
            coin_size = state.mesh.coin_size
            size = state.mesh.size
            num_particles = state.num_particles
            ind = ndim * num_particles
            expected_elems = size
            size_per_coin = int(coin_size / ndim)
            cs_size = size_per_coin * size
            dims = [size for p in range(ind)]

            if state.num_particles == 1:
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
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                a = []

                for p in range(num_particles):
                    a.append(m[0][p])

                a.append(m[1])

                return tuple(a)
        elif state.mesh.is_2d():
            ndim = state.mesh.dimension
            coin_size = state.mesh.coin_size
            size_x, size_y = state.mesh.size
            num_particles = state.num_particles
            ind = ndim * num_particles
            expected_elems = size_x * size_y
            size_per_coin = int(coin_size / ndim)
            cs_size_x = size_per_coin * size_x
            cs_size_y = size_per_coin * size_y
            cs_size_xy = cs_size_x * cs_size_y
            dims = []

            for p in range(0, ind, ndim):
                dims.append(state.mesh.size[0])
                dims.append(state.mesh.size[1])

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
            self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        expected_size = Utils.get_size_of_type(float) * expected_elems
        num_partitions = Utils.get_num_partitions(
            state.data.context, expected_size)

        data_type = state.data_type()

        rdd = state.data.filter(
            lambda m: m[1] != data_type
        ).map(
            __map
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        ).map(
            __unmap
        )

        pdf = JointPDF(rdd, shape, state.mesh,
                       state.num_particles).materialize(storage_level)

        self._logger.info("checking if the probabilities sum one...")

        round_precision = int(Utils.get_conf(
            self._spark_context, 'quantum.math.roundPrecision'))

        if round(pdf.sum_values(), round_precision) != 1.0:
            self._logger.error("PDFs must sum one")
            raise ValueError("PDFs must sum one")

        app_id = self._spark_context.applicationId

        if self._profiler is not None:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_pdf(
                'fullMeasurement', pdf, (datetime.now() - t1).total_seconds())

            self._logger.info(
                "full measurement was done in {}s".format(info['buildingTime']))
            self._logger.info(
                "PDF with full measurement is consuming {} bytes in memory and {} bytes in disk".format(
                    info['memoryUsed'], info['diskUsed']
                )
            )

            self._profiler.log_rdd(app_id=app_id)

        return pdf

    def measure_collision(self, state, system_measurement,
                          storage_level=StorageLevel.MEMORY_AND_DISK):
        """Filter the measurement of the entire system by checking when
        all particles are located at the same site of the mesh.

        Parameters
        ----------
        state : :py:class:`sparkquantum.dtqw.state.State`
            A system state.
        system_measurement : :py:class:`sparkquantum.math.statistics.pdf.PDF`
            The measurement of all possible positions of the entire system.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.math.statistics.collision_pdf.CollisionPDF`
            The PDF with all possible positions of the system when all particles are located at the same site.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid or if the system is composed by only one particle.

        TypeError
            If `system_measurement` is not valid.

        """
        if state.num_particles <= 1:
            self._logger.error(
                "the measurement of collision cannot be performed for quantum walks with only one particle")
            raise NotImplementedError(
                "the measurement of collision cannot be performed for quantum walks with only one particle")

        self._logger.info(
            "measuring the state of the system considering that the particles are at the same positions...")

        t1 = datetime.now()

        if not is_pdf(system_measurement):
            self._logger.error(
                "'PDF' instance expected, not '{}'".format(type(system_measurement)))
            raise TypeError("'PDF' instance expected, not '{}'".format(
                type(system_measurement)))

        if state.mesh.is_1d():
            ndim = state.mesh.dimension
            size = state.mesh.size
            num_particles = state.num_particles
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
        elif state.mesh.is_2d():
            ndim = state.mesh.dimension
            size_x, size_y = state.mesh.size
            num_particles = state.num_particles
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
            self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        expected_size = Utils.get_size_of_type(float) * expected_elems
        num_partitions = Utils.get_num_partitions(
            state.data.context, expected_size)

        rdd = system_measurement.data.filter(
            __filter
        ).map(
            __map
        ).coalesce(
            num_partitions
        )

        pdf = CollisionPDF(rdd, shape, state.mesh,
                           state.num_particles).materialize(storage_level)

        app_id = self._spark_context.applicationId

        if self._profiler is not None:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_pdf(
                'collisionMeasurement', pdf, (datetime.now() - t1).total_seconds())

            self._logger.info(
                "collision measurement was done in {}s".format(info['buildingTime']))
            self._logger.info(
                "PDF with collision measurement is consuming {} bytes in memory and {} bytes in disk".format(
                    info['memoryUsed'], info['diskUsed']
                )
            )

            self._profiler.log_rdd(app_id=app_id)

        return pdf

    def measure_particle(self, state, particle,
                         storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the partial measurement of a particle of the system state.

        Parameters
        ----------
        state : :py:class:`sparkquantum.dtqw.state.State`
            A system state.
        particle : int
            The desired particle to be measured. The particle number starts by 0.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.math.statistics.marginal_pdf.MarginalPDF`
            The PDF with all possible possitions of the desired particle.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If `particle` is not valid, i.e., particle number does not belong to the walk,
            if the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid or
            if the sum of the calculated PDF is not equal to one.

        """
        if particle < 0 or particle >= state.num_particles:
            self._logger.error("invalid particle number")
            raise ValueError("invalid particle number")

        self._logger.info(
            "measuring the state of the system for particle {}...".format(particle + 1))

        t1 = datetime.now()

        repr_format = int(Utils.get_conf(self._spark_context,
                                         'quantum.dtqw.state.representationFormat'))

        if state.mesh.is_1d():
            ndim = state.mesh.dimension
            coin_size = state.mesh.coin_size
            size = state.mesh.size
            num_particles = state.num_particles
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
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                return m
        elif state.mesh.is_2d():
            ndim = state.mesh.dimension
            coin_size = state.mesh.coin_size
            size_x, size_y = state.mesh.size
            num_particles = state.num_particles
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
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                return m[0][0], m[0][1], m[1]
        else:
            self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        expected_size = Utils.get_size_of_type(float) * expected_elems
        num_partitions = Utils.get_num_partitions(
            state.data.context, expected_size)

        data_type = state.data_type()

        rdd = state.data.filter(
            lambda m: m[1] != data_type
        ).map(
            __map
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        ).map(
            __unmap
        )

        pdf = MarginalPDF(rdd, shape, state.mesh,
                          state.num_particles).materialize(storage_level)

        self._logger.info("checking if the probabilities sum one...")

        round_precision = int(Utils.get_conf(
            self._spark_context, 'quantum.math.roundPrecision'))

        if round(pdf.sum_values(), round_precision) != 1.0:
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

    def measure_particles(
            self, state, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the partial measurement of each particle of the system state.

        Parameters
        ----------
        state : :py:class:`sparkquantum.dtqw.state.State`
            A system state.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        tuple
            A tuple containing the PDF with all possible positions of each particle.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If `particle` is not valid, i.e., particle number does not belong to the walk,
            if the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid or
            if the sum of the calculated PDF is not equal to one.

        """
        return [self.measure_particle(state, p, storage_level)
                for p in range(state.num_particles)]

    def measure(self, state, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the system state.

        If the state is composed by only one particle, the system measurement is performed and returned.
        In other cases, the measurement process will return a tuple containing
        the system measurement, the collision measurement - probabilities of each mesh site
        with all particles located at - and the partial measurement of each particle.

        Parameters
        ----------
        state : :py:class:`sparkquantum.dtqw.state.State`
            A system state.
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
        if state.num_particles == 1:
            return self.measure_system(state, storage_level)
        else:
            system_measurement = self.measure_system(state, storage_level)
            collision_measurement = self.measure_collision(
                state, system_measurement, storage_level)
            partial_measurements = self.measure_particles(state, storage_level)

            return system_measurement, collision_measurement, partial_measurements
