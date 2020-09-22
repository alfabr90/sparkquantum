import fileinput
import math
from datetime import datetime
from glob import glob

import numpy as np
from pyspark import RDD, SparkContext, StorageLevel

from sparkquantum import conf, constants, util
from sparkquantum.dtqw.interaction.interaction import is_interaction
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.dtqw.mesh.percolation.permanent import is_permanent
from sparkquantum.dtqw.operator import Operator
from sparkquantum.dtqw.particle import is_particle
from sparkquantum.dtqw.profiler import QuantumWalkProfiler
from sparkquantum.dtqw.state import State, is_state
from sparkquantum.math.matrix import Matrix

__all__ = ['DiscreteTimeQuantumWalk']


class DiscreteTimeQuantumWalk:
    """A representation of a discrete time quantum walk."""

    def __init__(self, mesh, interaction=None,
                 repr_format=constants.StateRepresentationFormatCoinPosition, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build a discrete time quantum walk object.

        Parameters
        ----------
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles are walking on.
        interaction : :py:class:`sparkquantum.dtqw.interaction.interaction.Interaction`, optional
            An interaction object. Default value is None.
        repr_format : int, optional
            Indicate how the quantum system will be represented.
            Default value is :py:const:`sparkquantum.constants.StateRepresentationFormatCoinPosition`.
        storage_level : :py:class:`pyspark.StorageLevel`, optional
            The desired storage level when materializing the operators' RDD.
            Default value is :py:const:`pyspark.StorageLevel.MEMORY_AND_DISK`.

        """
        if not is_mesh(mesh):
            raise TypeError(
                "'Mesh' instance expected, not '{}'".format(type(mesh)))

        if interaction is not None and not is_interaction(interaction):
            raise TypeError(
                "'Interaction' instance expected, not '{}'".format(type(interaction)))

        if (repr_format != constants.StateRepresentationFormatCoinPosition and
                repr_format != constants.StateRepresentationFormatPositionCoin):
            raise ValueError("invalid representation format")

        self._sc = SparkContext.getOrCreate()

        self._mesh = mesh
        self._interaction = interaction

        self._coin_operators = []
        self._shift_operator = None
        self._interaction_operator = None
        self._evolution_operators = []

        self._particles = []

        self._inistate = None
        self._curstate = None
        self._curstep = 0

        self._repr_format = repr_format
        self._storage_level = storage_level

        self._logger = util.get_logger(
            self._sc, self.__class__.__name__)
        self._profiler = QuantumWalkProfiler()

    @property
    def sc(self):
        """:py:class:`pyspark.SparkContext`"""
        return self._sc

    @property
    def mesh(self):
        """:py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`"""
        return self._mesh

    @property
    def interaction(self):
        """:py:class:`sparkquantum.dtqw.interaction.interaction.Interaction`"""
        return self._interaction

    @property
    def coin_operators(self):
        """tuple of :py:class:`sparkquantum.dtqw.operator.Operator`"""
        return tuple(self._coin_operators)

    @property
    def shift_operator(self):
        """:py:class:`sparkquantum.dtqw.operator.Operator`"""
        return self._shift_operator

    @property
    def interaction_operator(self):
        """:py:class:`sparkquantum.dtqw.operator.Operator`"""
        return self._interaction_operator

    @property
    def evolution_operators(self):
        """tuple of :py:class:`sparkquantum.dtqw.operator.Operator`"""
        return tuple(self._evolution_operators)

    @property
    def particles(self):
        """tuple of :py:class:`sparkquantum.dtqw.paticles.Particles`"""
        return tuple(self._particles)

    @property
    def inistate(self):
        """:py:class:`sparkquantum.dtqw.state.State`"""
        return self._inistate

    @property
    def curstate(self):
        """:py:class:`sparkquantum.dtqw.state.State`"""
        return self._curstate

    @property
    def curstep(self):
        """int"""
        return self._curstep

    @property
    def repr_format(self):
        """int"""
        return self._repr_format

    @property
    def storage_level(self):
        """:py:class:`pyspark.StorageLevel`"""
        return self._storage_level

    @property
    def profiler(self):
        """:py:class:`sparkquantum.dtqw.profiler.Profiler`."""
        return self._profiler

    def __del__(self):
        # In cases where multiple simulations are performed,
        # the same Python's logger object is used for the same class name.
        # Removing all its handlers, the object is reset.
        for h in self._logger.handlers:
            self._logger.removeHandler(h)

    def __str__(self):
        """Build a string representing this walk.

        Returns
        -------
        str
            The string representation of this walk.

        """
        particles = len(self._particles)

        if particles == 1:
            particles = '1 particle'
        else:
            if self._interaction:
                particles = '{} interacting particles by {}'.format(
                    particles, self._interaction)
            else:
                particles = '{} particles'.format(particles)

        return '{} with {} over a {}'.format(
            'Discrete time quantum walk', particles, self._mesh)

    def _profile_operator(self, profile_title, log_title, operator, time):
        app_id = self._sc.applicationId

        self._profiler.profile_resources(app_id)
        self._profiler.profile_executors(app_id)

        info = self._profiler.profile_operator(
            profile_title, operator, time)

        if info is not None:
            self._logger.info(
                "{} was built in {}s".format(log_title, info['buildingTime']))
            self._logger.info(
                "{} is consuming {} bytes in memory and {} bytes in disk".format(
                    log_title, info['memoryUsed'], info['diskUsed']))

        if conf.get(self._sc,
                    'sparkquantum.dtqw.profiler.logExecutors') == 'True':
            self._profiler.log_executors(app_id=app_id)

    def _profile_state(self, profile_title, log_title, state, time):
        app_id = self._sc.applicationId

        self._profiler.profile_resources(app_id)
        self._profiler.profile_executors(app_id)

        info = self._profiler.profile_state(profile_title, state, time)

        if info is not None:
            self._logger.info(
                "{} is consuming {} bytes in memory and {} bytes in disk".format(
                    log_title, info['memoryUsed'], info['diskUsed']
                )
            )

        if conf.get(self._sc,
                    'sparkquantum.dtqw.profiler.logExecutors') == 'True':
            self._profiler.log_executors(app_id=app_id)

    def _create_coin_operators(self):
        for p in range(len(self._particles)):
            particle = self._particles[p]

            if len(self._coin_operators) < p + 1:
                self._logger.info(
                    "building coin operator for particle {}...".format(p + 1)
                )

                time = datetime.now()

                co = self._particles[p].coin.create_operator(
                    self._mesh.sites, repr_format=self._repr_format
                ).change_coordinate(
                    constants.MatrixCoordinateMultiplicand
                )

                num_partitions = util.get_num_partitions(
                    self._sc,
                    util.get_size_of_type(co.dtype) * co.nelem
                )

                self._coin_operators.append(
                    co.partition_by(
                        num_partitions=num_partitions
                    ).materialize(self._storage_level)
                )

                time = (datetime.now() - time).total_seconds()

                self._profile_operator(
                    'coinOperator{}'.format(p + 1),
                    'coin operator for particle {}'.format(p + 1),
                    self._coin_operators[p],
                    time)

    def _create_shift_operator(self):
        self._logger.info("building shift operator...")

        time = datetime.now()

        so = self._mesh.create_operator(repr_format=self._repr_format).change_coordinate(
            constants.MatrixCoordinateMultiplier
        )

        num_partitions = util.get_num_partitions(
            self._sc,
            util.get_size_of_type(so.dtype) * so.nelem
        )

        self._shift_operator = so.partition_by(
            num_partitions=num_partitions).materialize(self._storage_level)

        time = (datetime.now() - time).total_seconds()

        self._profile_operator('shiftOperator', 'shift operator',
                               self._shift_operator, time)

    def _create_interaction_operator(self):
        self._logger.info("building interaction operator...")

        time = datetime.now()

        particles = len(self._particles)

        io = self._interaction.create_operator(self._mesh, particles, repr_format=self._repr_format).change_coordinate(
            constants.MatrixCoordinateMultiplier
        )

        partitions = util.get_num_partitions(
            self._sc,
            util.get_size_of_type(io.dtype) * io.nelem
        )

        self._interaction_operator = io.partition_by(
            num_partitions=partitions).materialize(self._storage_level)

        time = (datetime.now() - time).total_seconds()

        self._profile_operator('interactionOperator', 'interaction operator',
                               self._interaction_operator, time)

    def _create_evolution_operators(self):
        """Build the evolution operators for the walk.

        This method builds a list with n operators, where n is the number of particles of the system.
        In a multiparticle quantum walk, each operator is built by applying a tensor product between
        the evolution operator and ``n-1`` identity matrices as follows:

            ``W1 = U1 (X) I2 (X) ... (X) In
            Wi = I1 (X) ... (X) Ii-1 (X) Ui (X) Ii+1 (X) ... In
            Wn = I1 (X) ... (X) In-1 (X) Un``

        Raises
        ------
        ValueError
            If the chosen 'sparkquantum.dtqw.evolutionOperator.kroneckerMode' configuration is not valid.

        """
        self._logger.info("building evolution operators...")

        self._create_coin_operators()

        if self._shift_operator is None:
            self._logger.info(
                "no shift operator has been set. A new one will be built")

            self._create_shift_operator()

        self._destroy_evolution_operators()

        particles = len(self._particles)

        if particles > 1:
            kron_mode = conf.get(
                self._sc, 'sparkquantum.dtqw.evolutionOperator.kroneckerMode')

        for p in range(particles):
            self._logger.info(
                "building evolution operator for particle {}...".format(p + 1))

            time = datetime.now()

            eo = self._shift_operator.multiply(self._coin_operators[p])

            dtype = self._coin_operators[p].dtype
            nelem = eo.nelem * eo.shape[0] ** (particles - 1)

            shape = eo.shape

            if particles > 1:
                shape_tmp = shape

                if kron_mode == constants.KroneckerModeBroadcast:
                    eo_broad = util.broadcast(
                        self._sc, eo.data.collect()
                    )
                elif kron_mode == constants.KroneckerModeDump:
                    path = util.get_temp_path(
                        conf.get(self._sc,
                                 'sparkquantum.dtqw.evolutionOperator.tempPath')
                    )

                    eo.dump(path)
                else:
                    self._logger.error("invalid kronecker mode")
                    raise ValueError("invalid kronecker mode")

                if p == 0:
                    # The first particle's evolution operator consists in applying the tensor product between the
                    # evolution operator and the other particles' corresponding identity matrices
                    #
                    # W1 = U1 (X) I2 (X) ... (X) In
                    rdd_shape = (
                        shape_tmp[0] ** (particles - 1 - p),
                        shape_tmp[1] ** (particles - 1 - p)
                    )

                    if kron_mode == constants.KroneckerModeBroadcast:
                        def __map(m):
                            for i in eo_broad.value:
                                yield i[0] * rdd_shape[0] + m, i[1] * rdd_shape[1] + m, i[2]
                    elif kron_mode == constants.KroneckerModeDump:
                        def __map(m):
                            with fileinput.input(files=glob(path + '/part-*')) as f:
                                for line in f:
                                    l = line.split()
                                    yield int(l[0]) * rdd_shape[0] + m, int(l[1]) * rdd_shape[1] + m, complex(l[2])

                    rdd = self._sc.range(
                        rdd_shape[0]
                    ).flatMap(
                        __map
                    )

                    shape = (rdd_shape[0] * shape_tmp[0],
                             rdd_shape[1] * shape_tmp[1])
                else:
                    time = datetime.now()

                    # For the other particles, each one has its operator built by applying the
                    # tensor product between its previous particles' identity matrices and its evolution operator.
                    #
                    # Wi = I1 (X) ... (X) Ii-1 (X) Wi ...
                    rdd_shape = (
                        shape_tmp[0] ** p,
                        shape_tmp[1] ** p
                    )

                    if kron_mode == constants.KroneckerModeBroadcast:
                        def __map(m):
                            for i in eo_broad.value:
                                yield m * shape_tmp[0] + i[0], m * shape_tmp[1] + i[1], i[2]
                    elif kron_mode == constants.KroneckerModeDump:
                        def __map(m):
                            with fileinput.input(files=glob(path + '/part-*')) as f:
                                for line in f:
                                    l = line.split()
                                    yield m * shape_tmp[0] + int(l[0]), m * shape_tmp[1] + int(l[1]), complex(l[2])

                    rdd = self._sc.range(
                        rdd_shape[0]
                    ).flatMap(
                        __map
                    )

                    shape = (rdd_shape[0] * shape_tmp[0],
                             rdd_shape[1] * shape_tmp[1])

                    # Then, the tensor product is applied between the following particles' identity matrices.
                    #
                    # ... (X) Ii+1 (X) ... In
                    #
                    # If it is the last particle, the tensor product is applied between
                    # the pre-identity and evolution operators
                    #
                    # ... (X) Ii-1 (X) Wn
                    if p < particles - 1:
                        rdd_shape = (
                            shape_tmp[0] ** (particles - 1 - p),
                            shape_tmp[1] ** (particles - 1 - p)
                        )

                        def __map(m):
                            for i in range(rdd_shape[0]):
                                yield m[0] * rdd_shape[0] + i, m[1] * rdd_shape[1] + i, m[2]

                        rdd = rdd.flatMap(
                            __map
                        )

                        shape = (rdd_shape[0] * shape_tmp[0],
                                 rdd_shape[1] * shape_tmp[1])

                eo = Operator(rdd, shape,
                              dtype=dtype, nelem=nelem)

            num_partitions = util.get_num_partitions(
                self._sc,
                util.get_size_of_type(dtype) *
                nelem
            )

            eo = eo.change_coordinate(
                constants.MatrixCoordinateMultiplier
            ).partition_by(
                num_partitions=num_partitions
            ).persist(self._storage_level)

            if conf.get(
                    self._sc, 'sparkquantum.dtqw.evolutionOperator.checkpoint') == 'True':
                eo = eo.checkpoint()

            self._evolution_operators.append(
                eo.materialize(self._storage_level))

            time = (datetime.now() - time).total_seconds()

            self._profile_operator(
                'evolutionOperatorParticle{}'.format(p + 1),
                'evolution operator for particle {}'.format(p + 1),
                self._evolution_operators[-1],
                time)

            if particles > 1:
                if kron_mode == constants.KroneckerModeBroadcast:
                    eo_broad.unpersist()
                elif kron_mode == constants.KroneckerModeDump:
                    util.remove_path(path)

        self._shift_operator.unpersist()

    def _destroy_state(self):
        """Call the :py:func:`sparkquantum.dtqw.state.State.destroy` method."""
        if self._curstate is not None:
            self._curstate.destroy()
            self._curstate = None

    def _destroy_coin_operators(self):
        """Call the :py:func:`sparkquantum.dtqw.operator.Operator.destroy` method."""
        for co in self._coin_operators:
            co.destroy()
        self._coin_operators.clear()

    def _destroy_shift_operator(self):
        """Call the :py:func:`sparkquantum.dtqw.operator.Operator.destroy` method."""
        if self._shift_operator is not None:
            self._shift_operator.destroy()
            self._shift_operator = None

    def _destroy_interaction_operator(self):
        """Call the :py:func:`sparkquantum.dtqw.operator.Operator.destroy` method."""
        if self._interaction_operator is not None:
            self._interaction_operator.destroy()
            self._interaction_operator = None

    def _destroy_evolution_operators(self):
        """Call the :py:func:`sparkquantum.dtqw.operator.Operator.destroy` method."""
        for eo in self._evolution_operators:
            eo.destroy()
        self._evolution_operators.clear()

    def _destroy_operators(self):
        """Release all operators from memory and/or disk."""
        self._logger.info("destroying operators...")

        self._destroy_coin_operators()
        self._destroy_shift_operator()
        self._destroy_interaction_operator()
        self._destroy_evolution_operators()

        self._logger.info("operators have been destroyed")

    def _get_walk_configs(self):
        configs = {}

        configs['checkpointing_frequency'] = int(
            conf.get(
                self._sc, 'sparkquantum.dtqw.checkpointingFrequency'
            )
        )

        configs['dumping_frequency'] = int(
            conf.get(
                self._sc, 'sparkquantum.dtqw.dumpingFrequency'
            )
        )

        if configs['dumping_frequency'] >= 0:
            configs['dumping_path'] = conf.get(
                self._sc, 'sparkquantum.dumpingPath'
            )

            if not configs['dumping_path'].endswith('/'):
                configs['dumping_path'] += '/'

        configs['check_unitary'] = conf.get(
            self._sc, 'sparkquantum.dtqw.checkUnitary'
        )

        configs['dump_states_probability_distributions'] = conf.get(
            self._sc, 'sparkquantum.dtqw.dumpStatesProbabilityDistributions'
        )

        if configs['dump_states_probability_distributions'] == 'True':
            configs['dumping_path'] = conf.get(
                self._sc, 'sparkquantum.dumpingPath'
            )

            if not configs['dumping_path'].endswith('/'):
                configs['dumping_path'] += '/'

        return configs

    def add_particle(self, particle, cstate, position):
        """Add a particle to this quantum walk.

        Parameters
        ----------
        particle : :py:class:`sparkquantum.dtqw.particle.Particle`
            A particle to be present in the quantum walk.
        cstate : :py:class:`pyspark.RDD` or array-like
            The amplitudes of the coin state of the particle.
        position : int
            The particle's position (site number) over the mesh.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.DiscreteTimeQuantumWalk`
            A reference to this object.

        """
        if not is_particle(particle):
            self._logger.error(
                "'Particle' instance expected, not '{}'".format(type(particle)))
            raise TypeError(
                "'Particle' instance expected, not '{}'".format(type(particle)))

        if not self._mesh.has_site(position):
            self._logger.error("position out of mesh boundaries")
            raise ValueError("position out of mesh boundaries")

        if self._curstep > 0:
            self._logger.error(
                "it is not possible to add particles in a quantum walk that has been started")
            raise NotImplementedError(
                "it is not possible to add particles in a quantum walk that has been started")

        self.reset()
        self._destroy_interaction_operator()

        ndim = self._mesh.ndim
        cspace = 2 ** ndim
        pspace = self._mesh.sites

        shape = (cspace * pspace, 1)

        if isinstance(cstate, RDD):
            if self._repr_format == constants.StateRepresentationFormatCoinPosition:
                rdd = cstate.map(
                    lambda m: (m[0] * pspace + position, 0, m[-1])
                )
            elif self._repr_format == constants.StateRepresentationFormatPositionCoin:
                rdd = cstate.map(
                    lambda m: (position * cspace + m[0], 0, m[-1])
                )

            nelem = None
        elif isinstance(cstate, np.ndarray):
            if (len(cstate.shape) != 2 or
                    cstate.shape[0] != cstate or cstate.shape[1] != 1):
                self._logger.error("invalid coin state")
                raise ValueError("invalid coin state")

            rdd = self._sc.parallelize(
                ((nz, 0, cstate[nz][0]) for nz in cstate.nonzero()[0]))
            nelem = cstate.size
        else:
            nelem = len(cstate)

            if self._repr_format == constants.StateRepresentationFormatCoinPosition:
                data = (
                    (a * pspace + position, 0, cstate[a]) for a in range(nelem)
                )
            elif self._repr_format == constants.StateRepresentationFormatPositionCoin:
                data = (
                    (position * cspace + a, 0, cstate[a]) for a in range(nelem)
                )

            rdd = self._sc.parallelize(cstate)

        rdd = self._sc.parallelize(data)

        state = State(rdd, shape, self._mesh, (particle, ),
                      repr_format=self._repr_format, nelem=nelem)

        self._particles.append(particle)

        if self._inistate is None:
            self._inistate = state
        else:
            self._inistate = self._inistate.kron(state)

        return self

    def add_entanglement(self, particles, estate):
        """Add an entangled system of n particles to this quantum walk.

        Parameters
        ----------
        particles : tuple of :py:class:`sparkquantum.dtqw.particle.Particle`
            The entangled particles to be present in the quantum walk.
        estate : :py:class:`pyspark.RDD` or array-like
            The entangled state.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.DiscreteTimeQuantumWalk`
            A reference to this object.

        """
        for p in particles:
            if not is_particle(p):
                self._logger.error(
                    "'Particle' instance expected, not '{}'".format(type(p)))
                raise TypeError(
                    "'Particle' instance expected, not '{}'".format(type(p)))

        if self._curstep > 0:
            self._logger.error(
                "it is not possible to add particles in a quantum walk that has been started")
            raise NotImplementedError(
                "it is not possible to add particles in a quantum walk that has been started")

        self.reset()
        self._destroy_interaction_operator()

        ndim = self._mesh.ndim
        cspace = 2 ** ndim
        pspace = self._mesh.sites

        shape = ((cspace * pspace) ** len(particles), 1)

        if isinstance(estate, RDD):
            rdd = estate
            nelem = None
        elif isinstance(estate, np.ndarray):
            if len(estate.shape) != 2 or estate.shape[1] != 1:
                self._logger.error("invalid entangled state")
                raise ValueError("invalid entangled state")

            rdd = self._sc.parallelize(
                [(nz, 0, estate[nz][0]) for nz in estate.nonzero()[0]])
            nelem = estate.size
        else:
            rdd = self._sc.parallelize(((e[0], 0, e[-1]) for e in estate))
            nelem = len(estate)

        state = State(rdd, shape, self._mesh, particles,
                      repr_format=self._repr_format, nelem=nelem)

        self._particles.extend(particles)

        if self._inistate is None:
            self._inistate = state
        else:
            self._inistate = self._inistate.kron(state)

        return self

    def setup(self):
        """Setup this quantum walk, creating its operators when applicable.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.dtqw.DiscreteTimeQuantumWalk`
            A reference to this object.

        """
        self._profiler.log_rdd(app_id=self._sc.applicationId)

        if self._inistate is None:
            self._logger.error(
                "the initial state has not been created. Add some particles to this walk")
            raise ValueError(
                "the initial state has not been created. Add some particles to this walk")

        self._logger.info("checking if the initial state is unitary...")

        self._inistate.materialize(self._storage_level)

        if not self._inistate.is_unitary():
            self._logger.error("the initial state is not unitary")
            raise ValueError("the initial state is not unitary")

        self._curstate = self._inistate

        # Building evolution operators once if not simulating decoherence with
        # permanent percolations
        if (len(self._evolution_operators) == 0 and
                not is_permanent(self._mesh.percolation)):
            self._create_evolution_operators()

        if (len(self._particles) > 1 and
                self._interaction is not None and
                self._interaction_operator is None):
            self._create_interaction_operator()

    def destroy(self):
        """Clear the current state and all operators of this quantum walk.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.dtqw.DiscreteTimeQuantumWalk`
            A reference to this object.

        """
        self._inistate.unpersist()
        self._destroy_state()
        self._destroy_operators()

    def reset(self):
        """Reset this quantum walk.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.dtqw.DiscreteTimeQuantumWalk`
            A reference to this object.

        """
        if self._inistate is not None:
            self._inistate.unpersist()

        self._destroy_state()

        self._curstep = 0

        return self

    def step(self, configs):
        if self._curstate is None:
            self._logger.error("this quantum walk has not been setup")
            raise NotImplementedError("this quantum walk has not been setup")

        step = self._curstep + 1

        particles = len(self._particles)

        result = self._curstate

        # When there is a non-permanent percolations generator (e.g., random),
        # the evolution operators will be built in each step of the walk
        if (self._mesh.percolation is not None and
                not is_permanent(self._mesh.percolation)):
            self._destroy_shift_operator()
            self._create_evolution_operators()

        time = datetime.now()

        result_tmp = result.change_coordinate(
            constants.MatrixCoordinateMultiplicand
        )

        if particles > 1 and self._interaction_operator is not None:
            result_tmp = self._interaction_operator.multiply(result_tmp).change_coordinate(
                constants.MatrixCoordinateMultiplicand
            )

        for eo in self._evolution_operators:
            result_tmp = eo.multiply(result_tmp).change_coordinate(
                constants.MatrixCoordinateMultiplicand
            )

        result_tmp = result_tmp.change_coordinate(
            constants.MatrixCoordinateDefault
        )

        partitions = util.get_num_partitions(
            self._sc,
            util.get_size_of_type(result_tmp.dtype) * result_tmp.nelem
        )

        result_tmp.repartition(partitions).persist(self._storage_level)

        if configs['checkpointing_frequency'] >= 0 and step % configs['checkpointing_frequency'] == 0:
            result_tmp.checkpoint()

        result_tmp.materialize(self._storage_level)
        result.unpersist()

        time = (datetime.now() - time).total_seconds()

        self._logger.info(
            "step {} was done in {}s".format(step, time))

        self._profile_state(
            'systemState{}'.format(step),
            'system state after step {}'.format(step),
            result_tmp,
            time)

        if configs['dumping_frequency'] >= 0 and step % configs['dumping_frequency'] == 0:
            result_tmp.dump(
                configs['dumping_path'] + "states/" + str(step))

        if configs['check_unitary'] == 'True':
            if not result_tmp.is_unitary():
                self._logger.error(
                    "the state {} is not unitary".format(step))
                raise ValueError(
                    "the state {} is not unitary".format(step))

        if configs['dump_states_probability_distributions'] == 'True':
            if particles == 1:
                result_tmp.measure().dump(
                    configs['dumping_path'] + "probability_distributions/" + str(step))
            else:
                joint, collision, marginal = result_tmp.measure()

                joint.dump(
                    configs['dumping_path'] + "probability_distributions/joint/" + str(step))
                collision.dump(
                    configs['dumping_path'] + "probability_distributions/collision/" + str(step))
                for p in range(len(marginal)):
                    marginal[p].dump(
                        configs['dumping_path'] + "probability_distributions/marginal/" + str(step) + "/particle" + str(p + 1))

        app_id = self._sc.applicationId

        self._profiler.log_rdd(app_id=app_id)

        self._curstep += 1

        return result_tmp

    def walk(self, steps):
        """Perform the quantum walk.

        Parameters
        ----------
        steps : int
            The number of steps of the quantum walk.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.state.State`
            The final state of the system after performing the walk.

        Raises
        ------
        ValueError
            If the final state of the system is not unitary. This exception is also raised in cases
            where the 'sparkquantum.dtqw.walk.checkUnitary' configuration is set to 'True'
            and some of the intermediary states are not unitary
            or the 'sparkquantum.dtqw.walk.dumpStatesProbabilityDistribution' configuration is set to 'True'
            and the probability distribution of some of the intermediary states does not sum one.

        """
        self.reset()
        self.setup()

        configs = self._get_walk_configs()

        self._logger.info(
            "starting a {} for {} steps...".format(self, steps))

        time = datetime.now()

        step = self._curstep

        while step < steps:
            self._curstate = self.step(configs)
            step = self._curstep

        time = (datetime.now() - time).total_seconds()

        self._logger.info("walk was done in {}s".format(time))

        self._logger.info("checking if the final state is unitary...")

        if not self._curstate.is_unitary():
            self._logger.error("the final state is not unitary")
            raise ValueError("the final state is not unitary")

        # self._profile_state('finalState', 'final state', self._curstate, 0.0)

        return self._curstate
