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
from sparkquantum.dtqw.profiler import get_profiler
from sparkquantum.dtqw.state import State, is_state
from sparkquantum.math.matrix import Matrix

__all__ = ['DiscreteTimeQuantumWalk']


class DiscreteTimeQuantumWalk:
    """A representation of a discrete time quantum walk."""

    def __init__(self, mesh, interaction=None,
                 repr_format=constants.StateRepresentationFormatCoinPosition,
                 checkpoint_operators=False, storage_level=StorageLevel.MEMORY_AND_DISK):
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
        checkpoint_operators : bool, optional
            Indicate whether the evolution and interaction operators must be checkpointed.
            Default value is False.
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

        self._particles = []

        self._coin_operators = []
        self._shift_operator = None
        self._interaction_operator = None
        self._evolution_operators = []

        self._inistate = None
        self._curstate = None
        self._curstep = 0

        self._repr_format = repr_format
        self._checkpoint_operators = checkpoint_operators
        self._storage_level = storage_level

        self._logger = util.get_logger(
            self._sc, self.__class__.__name__)
        self._profiler = get_profiler()

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

    def _profile_operator(self, profile_title, operator, time):
        app_id = self._sc.applicationId

        self._profiler.profile_rdd(app_id)
        self._profiler.profile_executors(app_id)
        self._profiler.profile_operator(profile_title, operator, time)

    def _profile_state(self, profile_title, state, time):
        app_id = self._sc.applicationId

        self._profiler.profile_rdd(app_id)
        self._profiler.profile_executors(app_id)
        self._profiler.profile_state(profile_title, state, time)

    def _create_coin_operators(self):
        for i, particle in enumerate(self._particles, start=1):
            if len(self._coin_operators) < i:
                name = particle.name if particle.name is not None else 'unidentified'

                self._logger.info(
                    "building coin operator for particle {} ({})...".format(i, name))

                time = datetime.now()

                co = particle.coin.create_operator(
                    self._mesh.sites, repr_format=self._repr_format
                ).clear().to_coordinate(constants.MatrixCoordinateMultiplicand)

                num_partitions = util.get_num_partitions(
                    self._sc,
                    util.get_size_of_type(co.dtype) * co.nelem
                )

                co = co.partition_by(
                    num_partitions=num_partitions
                ).materialize(self._storage_level)

                self._coin_operators.append(co)

                time = (datetime.now() - time).total_seconds()

                self._logger.info(
                    "coin operator for particle {} ({}) was built in {}s".format(
                        i, name, time))

                self._profile_operator(
                    'coinOperatorParticle{}'.format(i), co, time)

    def _create_shift_operator(self):
        self._logger.info("building shift operator...")

        time = datetime.now()

        so = self._mesh.create_operator(
            repr_format=self._repr_format
        ).clear().to_coordinate(
            constants.MatrixCoordinateMultiplier
        )

        num_partitions = util.get_num_partitions(
            self._sc,
            util.get_size_of_type(so.dtype) * so.nelem
        )

        so = so.partition_by(
            num_partitions=num_partitions
        ).materialize(self._storage_level)

        self._shift_operator = so

        time = (datetime.now() - time).total_seconds()

        self._logger.info("shift operator was built in {}s".format(time))

        self._profile_operator('shiftOperator', so, time)

    def _create_interaction_operator(self):
        self._logger.info("building interaction operator...")

        time = datetime.now()

        particles = len(self._particles)

        io = self._interaction.create_operator(
            self._mesh, particles, repr_format=self._repr_format
        ).clear().to_coordinate(
            constants.MatrixCoordinateMultiplier
        )

        partitions = util.get_num_partitions(
            self._sc,
            util.get_size_of_type(io.dtype) * io.nelem
        )

        io = io.partition_by(
            num_partitions=partitions
        ).persist(self._storage_level)

        if self._checkpoint_operators:
            io = io.checkpoint()

        io = io.materialize(self._storage_level)

        self._interaction_operator = io

        time = (datetime.now() - time).total_seconds()

        self._logger.info("interaction operator was built in {}s".format(time))

        self._profile_operator('interactionOperator', io, time)

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

        for i, particle in enumerate(self._particles):
            name = particle.name if particle.name is not None else 'unidentified'

            self._logger.info(
                "building evolution operator for particle {} ({})...".format(i + 1, name))

            time = datetime.now()

            eo = self._shift_operator.multiply(self._coin_operators[i])

            dtype = self._coin_operators[i].dtype
            nelem = eo.nelem * eo.shape[0] ** (particles - 1)

            num_partitions = max(util.get_num_partitions(
                self._sc, util.get_size_of_type(dtype) * nelem
            ), eo.data.getNumPartitions())

            shape = eo.shape

            if particles > 1:
                shape_tmp = shape

                if i == 0:
                    # The first particle's evolution operator consists in applying the tensor product between the
                    # evolution operator and the other particles' corresponding identity matrices
                    #
                    # W1 = U1 (X) I2 (X) ... (X) In
                    rdd_shape = (
                        shape_tmp[0] ** (particles - 1 - i),
                        shape_tmp[1] ** (particles - 1 - i)
                    )

                    def __map(m):
                        for i in range(rdd_shape[0]):
                            yield m[0] * rdd_shape[0] + i, m[1] * rdd_shape[1] + i, m[2]

                    rdd = eo.data.flatMap(
                        __map
                    )
                else:
                    # For the other particles, each one has its operator built by applying the
                    # tensor product between its previous particles' identity matrices and its evolution operator.
                    #
                    # Wi = I1 (X) ... (X) Ii-1 (X) Wi ...
                    rdd_shape = (
                        shape_tmp[0] ** i,
                        shape_tmp[1] ** i
                    )

                    def __map(m):
                        for i in range(rdd_shape[0]):
                            yield i * shape_tmp[0] + m[0], i * shape_tmp[1] + m[1], m[2]

                    rdd = eo.data.flatMap(
                        __map
                    )

                    # Then, the tensor product is applied between the following particles' identity matrices.
                    #
                    # ... (X) Ii+1 (X) ... In
                    #
                    # If it is the last particle, the tensor product is applied between
                    # the pre-identity and evolution operators
                    #
                    # ... (X) Ii-1 (X) Wn
                    if i < particles - 1:
                        rdd_shape = (
                            shape_tmp[0] ** (particles - 1 - i),
                            shape_tmp[1] ** (particles - 1 - i)
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

            eo = eo.to_coordinate(
                constants.MatrixCoordinateMultiplier
            ).partition_by(
                num_partitions=num_partitions
            ).persist(self._storage_level)

            if self._checkpoint_operators:
                eo = eo.checkpoint()

            eo = eo.materialize(self._storage_level)

            self._evolution_operators.append(eo)

            time = (datetime.now() - time).total_seconds()

            self._logger.info(
                "evolution operator for particle {} ({}) was built in {}s".format(i + 1, name, time))

            self._profile_operator(
                'evolutionOperatorParticle{}'.format(i + 1), eo, time)

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
                data = ((c[0] * pspace + position, 0, c[-1]) for c in cstate)
            elif self._repr_format == constants.StateRepresentationFormatPositionCoin:
                data = ((position * cspace + c[0], 0, c[-1]) for c in cstate)

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

        if not self._inistate.is_unitary():
            self._logger.error("the initial state is not unitary")
            raise ValueError("the initial state is not unitary")

        self._curstate = self._inistate

        # Building evolution operators once if not simulating decoherence or
        # simulating decoherence with permanent percolations
        if (len(self._evolution_operators) == 0 and
                (self._mesh.percolation is None or
                    is_permanent(self._mesh.percolation))):
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
        if self._inistate is not None:
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

    def step(self, checkpoint_state=False):
        """Perform a step of this quantum walk.

        Parameters
        ----------
        checkpoint_state : bool, optional
            Indicate whether the state must be checkpointed.
            Default value is False.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.state.State`
            The state of the system after performing the step.

        Raises
        ------
        NotImplementedError
            If this quantum walk has not been setup.

        """
        if self._curstate is None:
            self._logger.error("this quantum walk has not been setup")
            raise NotImplementedError("this quantum walk has not been setup")

        step = self._curstep + 1

        result = self._curstate

        # When there is a non-permanent percolations generator (e.g., random),
        # the evolution operators will be built in each step of the walk
        if (self._mesh.percolation is not None and
                not is_permanent(self._mesh.percolation)):
            self._destroy_shift_operator()
            self._create_evolution_operators()

        time = datetime.now()

        if self._interaction_operator is not None:
            result = self._interaction_operator.multiply(
                result.to_coordinate(
                    constants.MatrixCoordinateMultiplicand
                ).partition_by(
                    self._interaction_operator.data.getNumPartitions()
                )
            )

        for eo in self._evolution_operators:
            result = eo.multiply(
                result.to_coordinate(
                    constants.MatrixCoordinateMultiplicand
                ).partition_by(
                    eo.data.getNumPartitions()
                )
            )

        result = result.persist(self._storage_level)

        if checkpoint_state:
            result = result.checkpoint()

        result = result.materialize(self._storage_level)

        self._curstate.unpersist()

        time = (datetime.now() - time).total_seconds()

        self._logger.info(
            "system state after step  {} was done in {}s".format(step, time))

        self._profile_state('systemState{}'.format(step), result, time)

        self._profiler.log_rdd(app_id=self._sc.applicationId)

        self._curstep += 1

        return result

    def walk(self, steps, checkpoint_frequency=None):
        """Perform the quantum walk.

        Parameters
        ----------
        steps : int
            The number of steps of the quantum walk.
        checkpoint_frequency : int, optional
            The rate which states will be checkpointed. Must be a positive value.
            When it is set to None, zero or negative values, the checkpointing does not occur.
            Default value is None.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.state.State`
            The final state of the system after performing the walk.

        Raises
        ------
        ValueError
            If the final state of the system is not unitary.

        """
        self.reset()
        self.setup()

        self._logger.info(
            "starting a {} for {} steps...".format(self, steps))

        time = datetime.now()

        step = self._curstep

        while step < steps:
            if (checkpoint_frequency is not None and
                    checkpoint_frequency > 0 and
                    (step + 1) % checkpoint_frequency == 0):
                checkpoint = True
            else:
                checkpoint = False

            self._curstate = self.step(checkpoint_state=checkpoint)
            step = self._curstep

        time = (datetime.now() - time).total_seconds()

        self._logger.info("walk was done in {}s".format(time))

        self._logger.info("checking if the final state is unitary...")

        if not self._curstate.is_unitary():
            self._logger.error("the final state is not unitary")
            raise ValueError("the final state is not unitary")

        return self._curstate
