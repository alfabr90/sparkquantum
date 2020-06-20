import math
import fileinput
from glob import glob
from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum.dtqw.operator import Operator
from sparkquantum.dtqw.qw_profiler import QuantumWalkProfiler
from sparkquantum.dtqw.state import State, is_state
from sparkquantum.utils.utils import Utils

__all__ = ['DiscreteTimeQuantumWalk']


class DiscreteTimeQuantumWalk:
    """Build the necessary operators and perform a discrete time quantum walk."""

    def __init__(self, initial_state,
                 storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build a discrete time quantum walk object.

        Parameters
        ----------
        initial_state : :py:class:`sparkquantum.dtqw.state.State`
            The initial state of the system.
        storage_level : :py:class:`pyspark.StorageLevel`, optional
            The desired storage level when materializing the operators' RDD.
            Default value is :py:const:`pyspark.StorageLevel.MEMORY_AND_DISK`.

        """
        self._spark_context = SparkContext.getOrCreate()
        self._initial_state = initial_state
        self._coin = self._initial_state.coin
        self._mesh = self._initial_state.mesh
        self._num_particles = self._initial_state.num_particles

        self._interaction = self._initial_state.interaction

        self._storage_level = storage_level

        self._coin_operator = None
        self._shift_operator = None
        self._interaction_operator = None
        self._walk_operators = None

        self._logger = Utils.get_logger(
            self._spark_context, self.__class__.__name__)
        self._profiler = QuantumWalkProfiler()

        if not is_state(self._initial_state):
            self._logger.error(
                "'State' instance expected, not '{}'".format(type(self._coin)))
            raise TypeError(
                "'State' instance expected, not '{}'".format(type(self._coin)))

    @property
    def spark_context(self):
        """:py:class:`pyspark.SparkContext`"""
        return self._spark_context

    @property
    def initial_state(self):
        """:py:class:`sparkquantum.dtqw.state.State`"""
        return self._initial_state

    @property
    def coin_operator(self):
        """:py:class:`sparkquantum.dtqw.operator.Operator`"""
        return self._coin_operator

    @property
    def shift_operator(self):
        """:py:class:`sparkquantum.dtqw.operator.Operator`"""
        return self._shift_operator

    @property
    def interaction_operator(self):
        """:py:class:`sparkquantum.dtqw.operator.Operator`"""
        return self._interaction_operator

    @property
    def walk_operators(self):
        """list of :py:class:`sparkquantum.dtqw.operator.Operator`"""
        return self._walk_operators

    @property
    def profiler(self):
        """:py:class:`sparkquantum.utils.profiler.Profiler`.

        To disable profiling, set it to None.

        """
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
        particles = '{} particle'.format(self._num_particles)

        if self._num_particles > 1:
            if self._interaction:
                particles = '{} interacting particles by {}'.format(
                    self._num_particles, self._interaction)
            else:
                particles = '{} particles'.format(self._num_particles)

        return '{} with {} and a {} over a {}'.format(
            'Discrete Time Quantum Walk', particles, self._coin, self._mesh)

    def _profile_operator(self, profile_title, log_title,
                          operator, initial_time):
        app_id = self._spark_context.applicationId

        self._profiler.profile_resources(app_id)
        self._profiler.profile_executors(app_id)

        info = self._profiler.profile_operator(
            profile_title,
            operator,
            (datetime.now() - initial_time).total_seconds())

        if info is not None:
            self._logger.info(
                "{} was built in {}s".format(
                    log_title, info['buildingTime']))
            self._logger.info(
                "{} is consuming {} bytes in memory and {} bytes in disk".format(
                    log_title, info['memoryUsed'], info['diskUsed']))

        if Utils.get_conf(self._spark_context,
                          'quantum.dtqw.profiler.logExecutors') == 'True':
            self._profiler.log_executors(app_id=app_id)

    def _create_coin_operator(self, storage_level):
        self._logger.info("building coin operator...")

        initial_time = datetime.now()

        co = self._coin.create_operator(self._mesh).change_coordinate(
            Utils.MatrixCoordinateMultiplicand
        )

        num_partitions = Utils.get_num_partitions(
            self._spark_context,
            Utils.get_size_of_type(co.data_type) * co.num_elements
        )

        self._coin_operator = co.partition_by(
            num_partitions=num_partitions).materialize(storage_level)

        self._profile_operator(
            'coinOperator',
            'coin operator',
            self._coin_operator,
            initial_time)

    def _create_shift_operator(self, storage_level):
        self._logger.info("building shift operator...")

        initial_time = datetime.now()

        so = self._mesh.create_operator().change_coordinate(
            Utils.MatrixCoordinateMultiplier
        )

        num_partitions = Utils.get_num_partitions(
            self._spark_context,
            Utils.get_size_of_type(so.data_type) * so.num_elements
        )

        self._shift_operator = so.partition_by(
            num_partitions=num_partitions).materialize(storage_level)

        self._profile_operator(
            'shiftOperator',
            'shift operator',
            self._shift_operator,
            initial_time)

    def _create_interaction_operator(self, storage_level):
        self._logger.info("building interaction operator...")

        initial_time = datetime.now()

        io = self._interaction.create_operator().change_coordinate(
            Utils.MatrixCoordinateMultiplier
        )

        num_partitions = Utils.get_num_partitions(
            self._spark_context,
            Utils.get_size_of_type(io.data_type) * io.num_elements
        )

        self._interaction_operator = io.partition_by(
            num_partitions=num_partitions).materialize(storage_level)

        self._profile_operator(
            'interactionOperator',
            'interaction operator',
            self._interaction_operator,
            initial_time)

    def _create_walk_operators(self, storage_level):
        """Build the walk operators for the walk.

        This method builds a list with n operators, where n is the number of particles of the system.
        In a multiparticle quantum walk, each operator is built by applying a tensor product between
        the evolution operator and ``n-1`` identity matrices as follows:

            ``W1 = W1 (X) I2 (X) ... (X) In
            Wi = I1 (X) ... (X) Ii-1 (X) Wi (X) Ii+1 (X) ... In
            Wn = I1 (X) ... (X) In-1 (X) Wn``

        Parameters
        ----------
        storage_level : :py:class:`pyspark.StorageLevel`, optional
            The desired storage level when materializing the RDD.

        Raises
        ------
        ValueError
            If the chosen 'quantum.dtqw.walkOperator.kroneckerMode' configuration is not valid.

        """
        app_id = self._spark_context.applicationId

        self._logger.info("building walk operator...")

        if self._coin_operator is None:
            self._logger.info(
                "no coin operator has been set. A new one will be built")

            self._create_coin_operator(storage_level)

        if self._shift_operator is None:
            self._logger.info(
                "no shift operator has been set. A new one will be built")

            self._create_shift_operator(storage_level)

        evolution_operator = self._shift_operator.multiply(
            self._coin_operator
        )

        initial_time = datetime.now()

        if self._num_particles == 1:
            eo = evolution_operator.change_coordinate(
                Utils.MatrixCoordinateMultiplier
            )

            num_partitions = Utils.get_num_partitions(
                self._spark_context,
                Utils.get_size_of_type(eo.data_type) * eo.num_elements
            )

            eo = eo.partition_by(
                num_partitions=num_partitions).persist(storage_level)

            if Utils.get_conf(self._spark_context,
                              'quantum.dtqw.walkOperator.checkpoint') == 'True':
                eo = eo.checkpoint()

            self._walk_operators = (eo.materialize(storage_level), )

            self._coin_operator.unpersist()
            self._shift_operator.unpersist()

            self._profile_operator(
                'walkOperator', 'walk operator', self._walk_operators[0], initial_time)
        else:
            shape = evolution_operator.shape
            shape_tmp = shape

            self._walk_operators = []

            kron_mode = Utils.get_conf(
                self._spark_context, 'quantum.dtqw.walkOperator.kroneckerMode')

            if kron_mode != Utils.KroneckerModeBroadcast and kron_mode != Utils.KroneckerModeDump:
                self._logger.error("invalid kronecker mode")
                raise ValueError("invalid kronecker mode")

            if kron_mode == Utils.KroneckerModeBroadcast:
                eo = Utils.broadcast(self._spark_context,
                                     evolution_operator.data.collect())
            elif kron_mode == Utils.KroneckerModeDump:
                path = Utils.get_temp_path(
                    Utils.get_conf(self._spark_context,
                                   'quantum.dtqw.walkOperator.tempPath')
                )

                evolution_operator.dump(path)

            self._coin_operator.unpersist()
            self._shift_operator.unpersist()

            for p in range(self._num_particles):
                self._logger.info(
                    "building walk operator for particle {}...".format(p + 1))

                if p == 0:
                    # The first particle's walk operator consists in applying the tensor product between the
                    # evolution operator and the other particles' corresponding identity matrices
                    #
                    # W1 = U (X) I2 (X) ... (X) In
                    rdd_shape = (
                        shape_tmp[0] ** (self._num_particles - 1 - p),
                        shape_tmp[1] ** (self._num_particles - 1 - p)
                    )

                    if kron_mode == Utils.KroneckerModeBroadcast:
                        def __map(m):
                            for i in eo.value:
                                yield i[0] * rdd_shape[0] + m, i[1] * rdd_shape[1] + m, i[2]
                    elif kron_mode == Utils.KroneckerModeDump:
                        def __map(m):
                            with fileinput.input(files=glob(path + '/part-*')) as f:
                                for line in f:
                                    l = line.split()
                                    yield int(l[0]) * rdd_shape[0] + m, int(l[1]) * rdd_shape[1] + m, complex(l[2])

                    rdd = self._spark_context.range(
                        rdd_shape[0]
                    ).flatMap(
                        __map
                    )

                    shape = (rdd_shape[0] * shape_tmp[0],
                             rdd_shape[1] * shape_tmp[1])
                else:
                    initial_time = datetime.now()

                    # For the other particles, each one has its operator built by applying the
                    # tensor product between its previous particles' identity matrices and its evolution operator.
                    #
                    # Wi = I1 (X) ... (X) Ii-1 (X) U ...
                    rdd_shape = (
                        shape_tmp[0] ** p,
                        shape_tmp[1] ** p
                    )

                    if kron_mode == Utils.KroneckerModeBroadcast:
                        def __map(m):
                            for i in eo.value:
                                yield m * shape_tmp[0] + i[0], m * shape_tmp[1] + i[1], i[2]
                    elif kron_mode == Utils.KroneckerModeDump:
                        def __map(m):
                            with fileinput.input(files=glob(path + '/part-*')) as f:
                                for line in f:
                                    l = line.split()
                                    yield m * shape_tmp[0] + int(l[0]), m * shape_tmp[1] + int(l[1]), complex(l[2])

                    rdd = self._spark_context.range(
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
                    # ... (X) Ii-1 (X) U
                    if p < self._num_particles - 1:
                        rdd_shape = (
                            shape_tmp[0] ** (self._num_particles - 1 - p),
                            shape_tmp[1] ** (self._num_particles - 1 - p)
                        )

                        def __map(m):
                            for i in range(rdd_shape[0]):
                                yield m[0] * rdd_shape[0] + i, m[1] * rdd_shape[1] + i, m[2]

                        rdd = rdd.flatMap(
                            __map
                        )

                        shape = (rdd_shape[0] * shape_tmp[0],
                                 rdd_shape[1] * shape_tmp[1])

                num_partitions = Utils.get_num_partitions(
                    self._spark_context,
                    Utils.get_size_of_type(evolution_operator.data_type) *
                    evolution_operator.num_elements *
                    evolution_operator.shape[0] ** (self._num_particles - 1)
                )

                wo = Operator(rdd, shape).change_coordinate(
                    Utils.MatrixCoordinateMultiplier
                ).partition_by(num_partitions=num_partitions).persist(storage_level)

                if Utils.get_conf(
                        self._spark_context, 'quantum.dtqw.walkOperator.checkpoint') == 'True':
                    wo = wo.checkpoint()

                self._walk_operators.append(wo.materialize(storage_level))

                self._profile_operator(
                    'walkOperatorParticle{}'.format(p + 1),
                    'walk operator for particle {}'.format(p + 1),
                    self._walk_operators[-1],
                    initial_time)

            eo.unpersist()

            if kron_mode == Utils.KroneckerModeDump:
                Utils.remove_path(path)

    def _destroy_coin_operator(self):
        """Call the :py:func:`sparkquantum.dtqw.operator.Operator.destroy` method."""
        if self._coin_operator is not None:
            self._coin_operator.destroy()
            self._coin_operator = None

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

    def _destroy_walk_operators(self):
        """Call the :py:func:`sparkquantum.dtqw.operator.Operator.destroy` method."""
        if self._walk_operators is not None:
            for wo in self._walk_operators:
                wo.destroy()
            self._walk_operators = None

    def destroy_operators(self):
        """Release all operators from memory and/or disk."""
        self._logger.info("destroying operators...")

        self._destroy_coin_operator()
        self._destroy_shift_operator()
        self._destroy_interaction_operator()
        self._destroy_walk_operators()

        self._logger.info("operators have been destroyed")

    def _get_walk_configs(self):
        configs = {}

        configs['checkpointing_frequency'] = int(
            Utils.get_conf(
                self._spark_context,
                'quantum.dtqw.walk.checkpointingFrequency'
            )
        )

        configs['dumping_frequency'] = int(
            Utils.get_conf(
                self._spark_context,
                'quantum.dtqw.walk.dumpingFrequency'
            )
        )

        if configs['dumping_frequency'] >= 0:
            configs['dumping_path'] = Utils.get_conf(
                self._spark_context,
                'quantum.dtqw.walk.dumpingPath'
            )

            if not configs['dumping_path'].endswith('/'):
                configs['dumping_path'] += '/'

        configs['check_unitary'] = Utils.get_conf(
            self._spark_context,
            'quantum.dtqw.walk.checkUnitary'
        )

        configs['dump_states_pdf'] = Utils.get_conf(
            self._spark_context,
            'quantum.dtqw.walk.dumpStatesPDF'
        )

        if configs['dump_states_pdf'] == 'True':
            configs['dumping_path'] = Utils.get_conf(
                self._spark_context,
                'quantum.dtqw.walk.dumpingPath'
            )

            if not configs['dumping_path'].endswith('/'):
                configs['dumping_path'] += '/'

        return configs

    def walk(self, steps):
        """Perform a walk.

        Parameters
        ----------
        steps : int
            The number of steps of the walk.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.state.State`
            The final state of the system after performing the walk.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If `steps` is not valid or if the collision phase is not valid.
            If the chosen 'quantum.dtqw.walkOperator.kroneckerMode' or 'quantum.dtqw.state.representationFormat' configuration is not valid.
            If operators' shapes are incompatible for multiplication.
            If the final state of the system is not unitary. This exception is also raised in cases
            where the 'quantum.dtqw.walk.checkUnitary' configuration is set to 'True' and some of the intermediary states are not unitary or
            the 'quantum.dtqw.walk.dumpStatesPDF' configuration is set to 'True' and the PDF of some of the intermediary states does not sum one.

        """
        if steps <= 0:
            self._logger.error(
                "the number of steps must be greater than or equal to 0")
            raise ValueError(
                "the number of steps must be greater than or equal to 0")

        if not self._mesh.check_steps(steps):
            self._logger.error(
                "invalid number of steps for the chosen mesh")
            raise ValueError("invalid number of steps for the chosen mesh")

        result = self._initial_state.materialize(self._storage_level)

        app_id = self._spark_context.applicationId

        self._profiler.profile_resources(app_id)
        self._profiler.profile_executors(app_id)

        info = self._profiler.profile_state('initialState', result, 0.0)

        if info is not None:
            self._logger.info(
                "initial state is consuming {} bytes in memory and {} bytes in disk".format(
                    info['memoryUsed'], info['diskUsed']
                )
            )

        self._profiler.log_rdd(app_id=app_id)

        configs = self._get_walk_configs()

        t1 = datetime.now()

        self._logger.info(
            "starting a {} for {} steps...".format(self, steps))

        # Building walk operators once if not simulating decoherence with
        # constant broken links
        if not self._mesh.broken_links or self._mesh.broken_links.is_constant():
            self._create_walk_operators(self._storage_level)

        if self._num_particles > 1 and self._interaction is not None and self._interaction_operator is None:
            self._create_interaction_operator(self._storage_level)

        for i in range(1, steps + 1, 1):
            # When there is a non-constant broken links (e.g., random),
            # the walk operators will be built in each step of the walk
            if self._mesh.broken_links and not self._mesh.broken_links.is_constant():
                self._destroy_shift_operator()
                self._destroy_walk_operators()
                self._create_walk_operators(self._storage_level)

            t_tmp = datetime.now()

            result_tmp = result.change_coordinate(
                Utils.MatrixCoordinateMultiplicand
            )

            if self._num_particles == 1:
                result_tmp = self._walk_operators[0].multiply(result_tmp)
            else:
                if self._interaction_operator is not None:
                    result_tmp = self._interaction_operator.multiply(result_tmp).change_coordinate(
                        Utils.MatrixCoordinateMultiplicand
                    )

                for wo in reversed(self._walk_operators):
                    result_tmp = wo.multiply(result_tmp).change_coordinate(
                        Utils.MatrixCoordinateMultiplicand
                    )

                result_tmp = result_tmp.change_coordinate(
                    Utils.MatrixCoordinateDefault
                )

            num_partitions = Utils.get_num_partitions(
                self._spark_context,
                Utils.get_size_of_type(
                    result_tmp.data_type) * result_tmp.num_elements
            )

            result_tmp.repartition(num_partitions).persist(self._storage_level)

            if configs['checkpointing_frequency'] >= 0 and i % configs['checkpointing_frequency'] == 0:
                result_tmp.checkpoint()

            result_tmp.materialize(self._storage_level)
            result.unpersist()

            if configs['dumping_frequency'] >= 0 and i % configs['dumping_frequency'] == 0:
                result_tmp.dump(
                    configs['dumping_path'] + "states/" + str(i))

            if configs['check_unitary'] == 'True':
                if not result_tmp.is_unitary():
                    self._logger.error(
                        "the state {} is not unitary".format(i))
                    raise ValueError(
                        "the state {} is not unitary".format(i))

            if configs['dump_states_pdf'] == 'True':
                if self._num_particles == 1:
                    result_tmp.measure().dump(
                        configs['dumping_path'] + "pdf/" + str(i))
                else:
                    joint, collision, marginal = result_tmp.measure()

                    joint.dump(
                        configs['dumping_path'] + "pdf/joint/" + str(i))
                    collision.dump(
                        configs['dumping_path'] + "pdf/collision/" + str(i))
                    for p in range(len(marginal)):
                        marginal[p].dump(
                            configs['dumping_path'] + "pdf/marginal/" + str(i) + "/particle" + str(p + 1))

            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_state(
                'systemState{}'.format(
                    i), result, (datetime.now() - t_tmp).total_seconds()
            )

            if info is not None:
                self._logger.info(
                    "step {} was done in {}s".format(i, info['buildingTime']))
                self._logger.info(
                    "system state of step {} is consuming {} bytes in memory and {} bytes in disk".format(
                        i, info['memoryUsed'], info['diskUsed']
                    )
                )

            self._profiler.log_rdd(app_id=app_id)

            result = result_tmp

        self._logger.info("walk was done in {}s".format(
            (datetime.now() - t1).total_seconds()))

        t1 = datetime.now()

        self._logger.info("checking if the final state is unitary...")

        if not result.is_unitary():
            self._logger.error("the final state is not unitary")
            raise ValueError("the final state is not unitary")

        self._logger.debug("unitarity check was done in {}s".format(
            (datetime.now() - t1).total_seconds()))

        self._profiler.profile_resources(app_id)
        self._profiler.profile_executors(app_id)

        info = self._profiler.profile_state('finalState', result, 0.0)

        if info is not None:
            self._logger.info(
                "final state is consuming {} bytes in memory and {} bytes in disk".format(
                    info['memoryUsed'], info['diskUsed']
                )
            )

        if Utils.get_conf(self._spark_context,
                          'quantum.dtqw.profiler.logExecutors') == 'True':
            self._profiler.log_executors(app_id=app_id)

        return result
