from datetime import datetime

from pyspark import StorageLevel

from sparkquantum import conf, constants, util
from sparkquantum.dtqw.state import is_state
from sparkquantum.dtqw.observable.observable import Observable
from sparkquantum.dtqw.particle import is_particle
from sparkquantum.math import util as mathutil
from sparkquantum.math.distribution import ProbabilityDistribution

__all__ = ['Position']


class Position(Observable):
    """Class for observables of particle's positions."""

    def __init__(self):
        """Build an observable object of particle's positions."""
        super().__init__()

    def measure_system(self, state,
                       storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the entire system state.

        Parameters
        ----------
        state : :py:class:`sparkquantum.dtqw.state.State`
            A system state.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.math.distribution.ProbabilityDistribution`
            The probability distribution regarding the possible positions of all particles of the quantum system state.

        Raises
        ------
        NotImplementedError
            If the state's coordinate format is not :py:const:`sparkquantum.constants.MatrixCoordinateDefault` or
            if the dimension of the mesh is not valid.

        ValueError
            If the chosen 'sparkquantum.dtqw.stateRepresentationFormat' configuration
            is not valid or if the sum of the calculated probability distribution is not equal to one.

        """
        if not is_state(state):
            self._logger.error(
                "'State' instance expected, not '{}'".format(type(self._state)))
            raise TypeError(
                "'State' instance expected, not '{}'".format(type(self._state)))

        if state.coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        self._logger.info("measuring the state of the system...")

        time = datetime.now()

        ndim = state.mesh.ndim
        particles = len(state.particles)
        csubspace = 2
        cspace = csubspace ** ndim
        psubspace = state.mesh.shape
        pspace = state.mesh.sites
        cpspace = cspace * pspace
        repr_format = state.repr_format

        ind = particles * ndim

        shape = (pspace, particles * ndim + 1)

        nelem = shape[0] * shape[1]

        if ndim == 1:
            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(m):
                    x = []

                    for p in range(particles):
                        x.append(
                            int(m[0] / (cpspace ** (particles - 1 - p))) % pspace)

                    return tuple(x), (abs(m[2]) ** 2).real
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(m):
                    x = []

                    for p in range(particles):
                        x.append(
                            int(m[0] / (cpspace ** (particles - 1 - p) * csubspace)) % pspace)

                    return tuple(x), (abs(m[2]) ** 2).real
            else:
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                x = []

                for p in range(particles):
                    x.append(m[0][p])

                x.append(m[1])

                return tuple(x)
        elif ndim == 2:
            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(m):
                    xy = []

                    for p in range(particles):
                        xy.append(
                            int(m[0] / (cpspace ** (particles - 1 - p) * psubspace[1])) % psubspace[0])
                        xy.append(
                            int(m[0] / (cpspace ** (particles - 1 - p))) % psubspace[1])

                    return tuple(xy), (abs(m[2]) ** 2).real
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(m):
                    xy = []

                    for p in range(particles):
                        xy.append(
                            int(m[0] / (cpspace ** (particles - 1 - p) * cspace * psubspace[1])) % psubspace[0])
                        xy.append(
                            int(m[0] / (cpspace ** (particles - 1 - p) * cspace)) % psubspace[1])

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

        expected_size = util.get_size_of_type(float) * nelem
        num_partitions = util.get_num_partitions(
            state.data.context, expected_size)

        rdd = state.data.map(
            __map
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        ).map(
            __unmap
        )

        distribution = ProbabilityDistribution(
            rdd, shape, state.mesh.axis(), nelem=nelem
        ).materialize(storage_level)

        self._logger.info("checking if the probabilities sum one...")

        round_precision = int(conf.get(
            self._sc, 'sparkquantum.math.roundPrecision'))

        if round(distribution.sum(), round_precision) != 1.0:
            self._logger.error("probability distributions must sum one")
            raise ValueError("probability distributions must sum one")

        time = (datetime.now() - time).total_seconds()

        self._profile_distribution(
            'systemMeasurement', 'system measurement',
            distribution, time)

        return distribution

    def measure_collision(self, state, system_measurement,
                          storage_level=StorageLevel.MEMORY_AND_DISK):
        """Filter the measurement of the entire system by checking when
        all particles are located at the same site of the mesh.

        Parameters
        ----------
        state : :py:class:`sparkquantum.dtqw.state.State`
            A system state.
        system_measurement : :py:class:`sparkquantum.dtqw.math.distribution.ProbabilityDistribution`
            The measurement of all possible positions of the entire system.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.math.distribution.ProbabilityDistribution`
            The probability distribution regarding the possible positions of all particles
            of the quantum system state when all particles are located at the same site.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid or if the system is composed by only one particle.

        """
        if not is_state(state):
            self._logger.error(
                "'State' instance expected, not '{}'".format(type(self._state)))
            raise TypeError(
                "'State' instance expected, not '{}'".format(type(self._state)))

        if len(state.particles) <= 1:
            self._logger.error(
                "the measurement of collision cannot be performed for quantum walks with only one particle")
            return None

        if not isinstance(system_measurement, ProbabilityDistribution):
            self._logger.error(
                "'PositionJointProbabilityDistribution' instance expected, not '{}'".format(type(system_measurement)))
            raise TypeError("'PositionJointProbabilityDistribution' instance expected, not '{}'".format(
                type(system_measurement)))

        self._logger.info(
            "measuring the state of the system considering the particles that are at the same positions...")

        time = datetime.now()

        ndim = state.mesh.ndim
        particles = len(state.particles)
        csubspace = 2
        cspace = csubspace ** ndim
        psubspace = state.mesh.shape
        pspace = state.mesh.sites
        cpspace = cspace * pspace
        repr_format = state.repr_format

        ind = particles * ndim

        shape = (pspace, ndim + 1)

        nelem = shape[0] * ndim

        variables = []

        if ndim == 1:
            def __filter(m):
                for p in range(particles):
                    if m[0] != m[p]:
                        return False
                return True

            def __map(m):
                return m[0], m[ind]
        elif ndim == 2:
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

        expected_size = util.get_size_of_type(float) * nelem
        num_partitions = util.get_num_partitions(
            state.data.context, expected_size)

        rdd = system_measurement.data.filter(
            __filter
        ).map(
            __map
        ).coalesce(
            num_partitions
        )

        distribution = ProbabilityDistribution(
            rdd, shape, state.mesh.axis(), nelem=nelem
        )

        norm = distribution.norm()

        rdd.map(
            lambda m: m[-1] / norm
        )

        distribution = ProbabilityDistribution(
            rdd, shape, state.mesh.axis(), nelem=nelem
        ).materialize(storage_level)

        time = (datetime.now() - time).total_seconds()

        self._profile_distribution(
            'collisionMeasurement', 'collision measurement',
            distribution, time)

        return distribution

    def measure_particle(self, state, particle,
                         storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the partial measurement of a particle of the system state.

        Parameters
        ----------
        state : :py:class:`sparkquantum.dtqw.state.State`
            A system state.
        particle : :py:class:`sparkquantum.dtqw.particle.Particle`
            The desired particle to be measured.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.math.distribution.ProbabilityDistribution`
            The probability distribution regarding the possible positions
            of a desired particle of the quantum system state.

        Raises
        ------
        NotImplementedError
            If the state's coordinate format is not :py:const:`sparkquantum.constants.MatrixCoordinateDefault` or
            if the dimension of the mesh is not valid.

        ValueError
            If `particle` is not valid, i.e., particle number does not belong to the walk,
            if the chosen 'sparkquantum.dtqw.stateRepresentationFormat' configuration is not valid or
            if the sum of the calculated probability distribution is not equal to one.

        """
        if not is_state(state):
            self._logger.error(
                "'State' instance expected, not '{}'".format(type(self._state)))
            raise TypeError(
                "'State' instance expected, not '{}'".format(type(self._state)))

        if not is_particle(particle):
            self._logger.error(
                "'Particle' instance expected, not '{}'".format(type(particle)))
            raise TypeError(
                "'Particle' instance expected, not '{}'".format(type(particle)))

        if state.coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        particles = len(state.particles)

        pind = None

        for p in range(particles):
            if state.particles[p].identifier == particle.identifier:
                pind = p
                break

        if pind is None:
            self._logger.error("particle not found")
            raise ValueError("particle not found")

        self._logger.info(
            "measuring the state of the system for particle {} ({})...".format(
                pind + 1,
                particle.identifier if particle.identifier is not None else 'unidentified'
            )
        )

        time = datetime.now()

        ndim = state.mesh.ndim
        csubspace = 2
        cspace = csubspace ** ndim
        psubspace = state.mesh.shape
        pspace = state.mesh.sites
        cpspace = cspace * pspace
        repr_format = state.repr_format

        shape = (pspace, ndim + 1)

        nelem = shape[0]

        if ndim == 1:
            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(m):
                    x = int(
                        m[0] / (cpspace ** (particles - 1 - pind))) % pspace
                    return x, (abs(m[2]) ** 2).real
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(m):
                    x = int(m[0] / (cpspace ** (particles -
                                                1 - pind) * csubspace)) % pspace
                    return x, (abs(m[2]) ** 2).real
            else:
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            def __unmap(m):
                return m
        elif ndim == 2:
            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(m):
                    xy = (
                        int(m[0] / (cpspace ** (particles -
                                                1 - pind) * psubspace[1])) % psubspace[0],
                        int(m[0] / (cpspace **
                                    (particles - 1 - pind))) % psubspace[1]
                    )
                    return xy, (abs(m[2]) ** 2).real
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(m):
                    xy = (
                        int(m[0] / (cpspace ** (particles - 1 -
                                                pind) * cspace * psubspace[1])) % psubspace[0],
                        int(m[0] / (cpspace ** (particles -
                                                1 - pind) * cspace)) % psubspace[1]
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

        expected_size = util.get_size_of_type(float) * nelem
        num_partitions = util.get_num_partitions(
            state.data.context, expected_size)

        rdd = state.data.map(
            __map
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        ).map(
            __unmap
        )

        distribution = ProbabilityDistribution(
            rdd, shape, state.mesh.axis(), nelem=nelem
        ).materialize(storage_level)

        self._logger.info("checking if the probabilities sum one...")

        round_precision = int(conf.get(
            self._sc, 'sparkquantum.math.roundPrecision'))

        if round(distribution.sum(), round_precision) != 1.0:
            self._logger.error("probability distributions must sum one")
            raise ValueError("probability distributions must sum one")

        time = (datetime.now() - time).total_seconds()

        self._profile_distribution(
            'partialMeasurementParticle{}'.format(pind + 1),
            'partial measurement for particle {} ({})...'.format(
                pind + 1,
                particle.identifier if particle.identifier is not None else 'unidentified'
            ),
            distribution, time)

        return distribution

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
            if the chosen 'sparkquantum.dtqw.stateRepresentationFormat' configuration is not valid or
            if the sum of the calculated probability distribution is not equal to one.

        """
        return [self.measure_particle(state, p, storage_level)
                for p in state.particles]

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
        :py:class:`sparkquantum.dtqw.math.distribution.ProbabilityDistribution` or tuple :py:class:`sparkquantum.dtqw.math.distribution.ProbabilityDistribution`
            If the system is composed by only one particle, tuple otherwise.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If the chosen 'sparkquantum.dtqw.stateRepresentationFormat' configuration is not valid or
            if the sum of the calculated probability distributions is not equal to one.

        """
        if len(state.particles) == 1:
            return self.measure_system(state, storage_level)
        else:
            joint = self.measure_system(state, storage_level)
            collision = self.measure_collision(state, joint, storage_level)
            marginal = self.measure_particles(state, storage_level)

            return joint, collision, marginal
