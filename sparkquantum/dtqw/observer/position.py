from datetime import datetime

from pyspark import StorageLevel

from sparkquantum import conf, constants, util
from sparkquantum.dtqw.state import is_state
from sparkquantum.dtqw.observer.observer import Observer
from sparkquantum.dtqw.particle import is_particle
from sparkquantum.math import util as mathutil
from sparkquantum.math.distribution import ProbabilityDistribution

__all__ = ['Position']


class Position(Observer):
    """Class for observers of particle's positions."""

    def __init__(self):
        """Build an observer object of particle's positions."""
        super().__init__()

    def _measure_system(self, state,
                        storage_level=StorageLevel.MEMORY_AND_DISK):
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

        self._profile_distribution('systemMeasurement', 'system measurement',
                                   distribution, time)

        return distribution

    def _measure_particle(self, state, particle,
                          storage_level=StorageLevel.MEMORY_AND_DISK):
        pind = None

        particles = len(state.particles)

        for p in range(particles):
            if state.particles[p] is particle:
                pind = p
                break

        if pind is None:
            self._logger.error("particle not found")
            raise ValueError("particle not found")

        name = particle.name if particle.name is not None else 'unidentified'

        self._logger.info(
            "measuring the state of the system for particle {} ({})...".format(pind + 1, name))

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
            'partial measurement for particle {} ({})'.format(pind + 1, name),
            distribution,
            time)

        return distribution

    def measure_collision(self, state, system_measurement,
                          storage_level=StorageLevel.MEMORY_AND_DISK):
        """Filter the measurement of the entire system by checking when
        all particles are located at the same site of the mesh.

        Notes
        -----
        Experimental

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
                "'ProbabilityDistribution' instance expected, not '{}'".format(type(system_measurement)))
            raise TypeError(
                "'ProbabilityDistribution' instance expected, not '{}'".format(type(system_measurement)))

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
        ).materialize(storage_level)

        time = (datetime.now() - time).total_seconds()

        self._profile_distribution('collisionMeasurement', 'collision measurement',
                                   distribution, time)

        return distribution

    def measure(self, state, particle=None,
                storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the system state.

        Parameters
        ----------
        state : :py:class:`sparkquantum.dtqw.state.State`
            The system state.
        particle : :py:class:`sparkquantum.dtqw.particle.Particle`
            The particle to have its position measured.
        storage_level : :py:class:`pyspark.StorageLevel`
            The desired storage level when materializing the RDD.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.math.distribution.ProbabilityDistribution`
            The probability distribution for positions.

        Raises
        ------
        NotImplementedError
            If the state's coordinate format is not :py:const:`sparkquantum.constants.MatrixCoordinateDefault` or
            if the dimension of the mesh is not valid.

        ValueError
            If `particle` is does not belong to the system state, if the state's 'sparkquantum.dtqw.stateRepresentationFormat'
            is not valid or if the sum of the calculated probability distributions is not equal to one.

        """
        if not is_state(state):
            self._logger.error(
                "'State' instance expected, not '{}'".format(type(self._state)))
            raise TypeError(
                "'State' instance expected, not '{}'".format(type(self._state)))

        if state.coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise NotImplementedError("invalid coordinate format")

        if particle is None:
            return self._measure_system(state,
                                        storage_level=storage_level)
        else:
            if not is_particle(particle):
                self._logger.error(
                    "'Particle' instance expected, not '{}'".format(type(particle)))
                raise TypeError(
                    "'Particle' instance expected, not '{}'".format(type(particle)))

            return self._measure_particle(state,
                                          particle=particle, storage_level=storage_level)
