from datetime import datetime

from pyspark import StorageLevel

from sparkquantum import conf, constants, util
from sparkquantum.dtqw.gauge.gauge import Gauge
from sparkquantum.dtqw.math.statistics.probability_distribution.position_collision_probability_distribution import PositionCollisionProbabilityDistribution
from sparkquantum.dtqw.math.statistics.probability_distribution.position_joint_probability_distribution import PositionJointProbabilityDistribution
from sparkquantum.dtqw.math.statistics.probability_distribution.position_marginal_probability_distribution import PositionMarginalProbabilityDistribution
from sparkquantum.dtqw.math.statistics.probability_distribution.position_probability_distribution import is_position_probability_distribution
from sparkquantum.math import util as mathutil

__all__ = ['PositionGauge']


class PositionGauge(Gauge):
    """Top-level class for system state's positions measurements (gauge)."""

    def __init__(self):
        """Build a top-level system state's positions measurement (gauge) object."""
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
        :py:class:`sparkquantum.dtqw.math.statistics.probability_distribution.position_join_probability_distribution.PositionJointProbabilityDistribution`
            The probability distribution regarding the possible positions of all particles of the quantum system state.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If the chosen 'sparkquantum.dtqw.state.representationFormat' configuration
            is not valid or if the sum of the calculated probability distribution is not equal to one.

        """
        self._logger.info("measuring the state of the system...")

        initial_time = datetime.now()

        repr_format = int(conf.get_conf(self._spark_context,
                                        'sparkquantum.dtqw.state.representationFormat'))

        if state.mesh.dimension == 1:
            ndim = state.mesh.dimension
            coin_size = state.mesh.coin_size
            size = state.mesh.size
            num_particles = state.num_particles
            ind = ndim * num_particles
            expected_elements = size
            size_per_coin = int(coin_size / ndim)
            cs_size = size_per_coin * size
            dims = [size for p in range(ind)]

            if state.num_particles == 1:
                dims.append(1)

            shape = tuple(dims)

            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(m):
                    x = []

                    for p in range(num_particles):
                        x.append(
                            int(m[0] / (cs_size ** (num_particles - 1 - p))) % size)

                    return tuple(x), (abs(m[2]) ** 2).real
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(m):
                    x = []

                    for p in range(num_particles):
                        x.append(
                            int(m[0] / (cs_size ** (num_particles - 1 - p) * size_per_coin)) % size)

                    return tuple(x), (abs(m[2]) ** 2).real
            else:
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                a = []

                for p in range(num_particles):
                    a.append(m[0][p])

                a.append(m[1])

                return tuple(a)
        elif state.mesh.dimension == 2:
            ndim = state.mesh.dimension
            coin_size = state.mesh.coin_size
            size_x, size_y = state.mesh.size
            num_particles = state.num_particles
            ind = ndim * num_particles
            expected_elements = size_x * size_y
            size_per_coin = int(coin_size / ndim)
            cs_size_x = size_per_coin * size_x
            cs_size_y = size_per_coin * size_y
            cs_size_xy = cs_size_x * cs_size_y
            dims = []

            for p in range(0, ind, ndim):
                dims.append(state.mesh.size[0])
                dims.append(state.mesh.size[1])

            shape = tuple(dims)

            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(m):
                    xy = []

                    for p in range(num_particles):
                        xy.append(
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p) * size_y)) % size_x)
                        xy.append(
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p))) % size_y)

                    return tuple(xy), (abs(m[2]) ** 2).real
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(m):
                    xy = []

                    for p in range(num_particles):
                        xy.append(
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p) * coin_size * size_y)) % size_x)
                        xy.append(
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p) * coin_size)) % size_y)

                    return tuple(xy), (abs(m[2]) ** 2).real
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

        expected_size = util.get_size_of_type(float) * expected_elements
        num_partitions = util.get_num_partitions(
            state.data.context, expected_size)

        rdd = mathutil.change_coordinate(
            mathutil.remove_zeros(
                state.data,
                state.data_type,
                state.coordinate_format),
            state.coordinate_format,
            constants.MatrixCoordinateDefault)

        rdd = rdd.map(
            __map
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        ).map(
            __unmap
        )

        probability_distribution = PositionJointProbabilityDistribution(
            rdd, shape, ndim * num_particles, state, num_elements=expected_elements
        ).materialize(storage_level)

        self._logger.info("checking if the probabilities sum one...")

        round_precision = int(conf.get_conf(
            self._spark_context, 'sparkquantum.math.roundPrecision'))

        if round(probability_distribution.sum(), round_precision) != 1.0:
            self._logger.error("Probability distributions must sum one")
            raise ValueError("Probability distributions must sum one")

        self._profile_probability_distribution(
            'systemMeasurement',
            'system measurement',
            probability_distribution,
            initial_time)

        return probability_distribution

    def measure_collision(self, state, system_measurement,
                          storage_level=StorageLevel.MEMORY_AND_DISK):
        """Filter the measurement of the entire system by checking when
        all particles are located at the same site of the mesh.

        Parameters
        ----------
        state : :py:class:`sparkquantum.dtqw.state.State`
            A system state.
        system_measurement : :py:class:`sparkquantum.dtqw.math.statistics.probability_distribution.position_joint_probability_distribution.PositionJointProbabilityDistribution`
            The measurement of all possible positions of the entire system.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.math.statistics.probability_distribution.position_collision_probability_distribution.PositionCollisionProbabilityDistribution`
            The probability distribution regarding the possible positions of all particles
            of the quantum system state when all particles are located at the same site.

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

        initial_time = datetime.now()

        if not isinstance(system_measurement,
                          PositionJointProbabilityDistribution):
            self._logger.error(
                "'PositionJointProbabilityDistribution' instance expected, not '{}'".format(type(system_measurement)))
            raise TypeError("'PositionJointProbabilityDistribution' instance expected, not '{}'".format(
                type(system_measurement)))

        if state.mesh.dimension == 1:
            ndim = state.mesh.dimension
            size = state.mesh.size
            num_particles = state.num_particles
            ind = ndim * num_particles
            expected_elements = size
            shape = (size, 1)

            def __filter(m):
                for p in range(num_particles):
                    if m[0] != m[p]:
                        return False
                return True

            def __map(m):
                return m[0], m[ind]
        elif state.mesh.dimension == 2:
            ndim = state.mesh.dimension
            size_x, size_y = state.mesh.size
            num_particles = state.num_particles
            ind = ndim * num_particles
            expected_elements = size_x * size_y
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

        expected_size = util.get_size_of_type(float) * expected_elements
        num_partitions = util.get_num_partitions(
            state.data.context, expected_size)

        rdd = system_measurement.data.filter(
            __filter
        ).map(
            __map
        ).coalesce(
            num_partitions
        )

        probability_distribution = PositionCollisionProbabilityDistribution(
            rdd, shape, ndim * num_particles, state, num_elements=expected_elements
        ).materialize(storage_level)

        self._profile_probability_distribution(
            'collisionMeasurement',
            'collision measurement',
            probability_distribution,
            initial_time)

        return probability_distribution

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
        :py:class:`sparkquantum.dtqw.math.statistics.probability_distribution.position_marginal_probability_distribution.PositionMarginalProbabilityDistribution`
            The probability distribution regarding the possible positions
            of a desired particle of the quantum system state.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If `particle` is not valid, i.e., particle number does not belong to the walk,
            if the chosen 'sparkquantum.dtqw.state.representationFormat' configuration is not valid or
            if the sum of the calculated probability distribution is not equal to one.

        """
        if particle < 0 or particle >= state.num_particles:
            self._logger.error("invalid particle number")
            raise ValueError("invalid particle number")

        self._logger.info(
            "measuring the state of the system for particle {}...".format(particle + 1))

        initial_time = datetime.now()

        repr_format = int(conf.get_conf(self._spark_context,
                                        'sparkquantum.dtqw.state.representationFormat'))

        if state.mesh.dimension == 1:
            ndim = state.mesh.dimension
            coin_size = state.mesh.coin_size
            size = state.mesh.size
            num_particles = state.num_particles
            expected_elements = size
            size_per_coin = int(coin_size / ndim)
            cs_size = size_per_coin * size
            shape = (size, 1)

            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(m):
                    x = int(
                        m[0] / (cs_size ** (num_particles - 1 - particle))) % size
                    return x, (abs(m[2]) ** 2).real
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(m):
                    x = int(m[0] / (cs_size ** (num_particles -
                                                1 - particle) * size_per_coin)) % size
                    return x, (abs(m[2]) ** 2).real
            else:
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                return m
        elif state.mesh.dimension == 2:
            ndim = state.mesh.dimension
            coin_size = state.mesh.coin_size
            size_x, size_y = state.mesh.size
            num_particles = state.num_particles
            expected_elements = size_x * size_y
            size_per_coin = int(coin_size / ndim)
            cs_size_x = size_per_coin * size_x
            cs_size_y = size_per_coin * size_y
            cs_size_xy = cs_size_x * cs_size_y
            shape = (size_x, size_y)

            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(m):
                    xy = (
                        int(m[0] / (cs_size_xy ** (num_particles -
                                                   1 - particle) * size_y)) % size_x,
                        int(m[0] / (cs_size_xy **
                                    (num_particles - 1 - particle))) % size_y
                    )
                    return xy, (abs(m[2]) ** 2).real
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(m):
                    xy = (
                        int(m[0] / (cs_size_xy ** (num_particles - 1 -
                                                   particle) * coin_size * size_y)) % size_x,
                        int(m[0] / (cs_size_xy ** (num_particles -
                                                   1 - particle) * coin_size)) % size_y
                    )
                    return xy, (abs(m[2]) ** 2).real
            else:
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                return m[0][0], m[0][1], m[1]
        else:
            self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        expected_size = util.get_size_of_type(float) * expected_elements
        num_partitions = util.get_num_partitions(
            state.data.context, expected_size)

        rdd = mathutil.change_coordinate(
            mathutil.remove_zeros(
                state.data,
                state.data_type,
                state.coordinate_format),
            state.coordinate_format,
            constants.MatrixCoordinateDefault)

        rdd = rdd.map(
            __map
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        ).map(
            __unmap
        )

        probability_distribution = PositionMarginalProbabilityDistribution(
            rdd, shape, state, num_elements=expected_elements
        ).materialize(storage_level)

        self._logger.info("checking if the probabilities sum one...")

        round_precision = int(conf.get_conf(
            self._spark_context, 'sparkquantum.math.roundPrecision'))

        if round(probability_distribution.sum(), round_precision) != 1.0:
            self._logger.error("Probability distributions must sum one")
            raise ValueError("Probability distributions must sum one")

        self._profile_probability_distribution(
            'partialMeasurementParticle{}'.format(
                particle + 1),
            'partial measurement for particle {}'.format(
                particle + 1),
            probability_distribution,
            initial_time)

        return probability_distribution

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
            A tuple containing the probability distribution with all possible positions of each particle.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If `particle` is not valid, i.e., particle number does not belong to the walk,
            if the chosen 'sparkquantum.dtqw.state.representationFormat' configuration is not valid or
            if the sum of the calculated probability distribution is not equal to one.

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
        :py:class:`sparkquantum.dtqw.math.statistics.probability_distribution.position_marginal_probability_distribution.PositionMarginalProbabilityDistribution` or tuple
            :py:class:`sparkquantum.dtqw.math.statistics.probability_distribution.position_marginal_probability_distribution.PositionMarginalProbabilityDistribution`
            if the system is composed by only one particle, tuple otherwise.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If the chosen 'sparkquantum.dtqw.state.representationFormat' configuration is not valid or
            if the sum of the calculated probability distributions is not equal to one.

        """
        if state.num_particles == 1:
            return self.measure_system(state, storage_level)
        else:
            system_measurement = self.measure_system(state, storage_level)
            collision_measurement = self.measure_collision(
                state, system_measurement, storage_level)
            partial_measurements = self.measure_particles(state, storage_level)

            return system_measurement, collision_measurement, partial_measurements
