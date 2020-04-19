import math
import fileinput
from glob import glob
from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum.dtqw.coin.coin import is_coin
from sparkquantum.dtqw.interaction.interaction import is_interaction
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.dtqw.operator import Operator, is_operator
from sparkquantum.dtqw.state import State
from sparkquantum.utils.logger import is_logger
from sparkquantum.utils.profiler import is_profiler
from sparkquantum.utils.utils import Utils

__all__ = ['DiscreteTimeQuantumWalk']


class DiscreteTimeQuantumWalk:
    """Build the necessary operators and perform a discrete time quantum walk."""

    def __init__(self, coin, mesh, num_particles, interaction=None):
        """Build a discrete time quantum walk object.

        Parameters
        ----------
        coin : :py:class:`sparkquantum.dtqw.coin.coin.Coin`
            The coin for the walk.
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles will walk over.
        num_particles : int
            The number of particles present in the walk.
        interaction : :py:class:`sparkquantum.dtqw.interaction.interaction.Interaction`, optional
            A particles interaction object.

        """
        self._spark_context = SparkContext.getOrCreate()
        self._coin = coin
        self._mesh = mesh
        self._num_particles = num_particles

        self._interaction = interaction

        self._coin_operator = None
        self._shift_operator = None
        self._interaction_operator = None
        self._walk_operators = None

        self._logger = None
        self._profiler = None

        self._validate()

    @property
    def spark_context(self):
        """:py:class:`pyspark.SparkContext`"""
        return self._spark_context

    @property
    def coin(self):
        """:py:class:`sparkquantum.dtqw.coin.coin.Coin`"""
        return self._coin

    @property
    def mesh(self):
        """:py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`"""
        return self._mesh

    @property
    def num_particles(self):
        """int"""
        return self._num_particles

    @property
    def interaction(self):
        """:py:class:`sparkquantum.dtqw.interaction.interaction.Interaction`"""
        return self._interaction

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
    def logger(self):
        """:py:class:`sparkquantum.utils.logger.Logger`.

        To disable logging, set it to None.

        """
        return self._logger

    @property
    def profiler(self):
        """:py:class:`sparkquantum.utils.profiler.Profiler`.

        To disable profiling, set it to None.

        """
        return self._profiler

    @logger.setter
    def logger(self, logger):
        if is_logger(logger) or logger is None:
            self._logger = logger
        else:
            raise TypeError(
                "'Logger' instance expected, not '{}'".format(type(logger)))

    @profiler.setter
    def profiler(self, profiler):
        if is_profiler(profiler) or profiler is None:
            self._profiler = profiler
        else:
            raise TypeError(
                "'Profiler' instance expected, not '{}'".format(type(profiler)))

    def _validate(self):
        if self._num_particles < 1:
            if self._logger is not None:
                self._logger.error("invalid number of particles")
            raise ValueError("invalid number of particles")

        if not is_coin(self._coin):
            if self._logger is not None:
                self._logger.error(
                    "'Coin' instance expected, not '{}'".format(type(self._coin)))
            raise TypeError(
                "'Coin' instance expected, not '{}'".format(type(self._coin)))

        if not is_mesh(self._mesh):
            if self._logger is not None:
                self._logger.error(
                    "'Mesh' instance expected, not '{}'".format(type(self._mesh)))
            raise TypeError(
                "'Mesh' instance expected, not '{}'".format(type(self._mesh)))

        if self._num_particles > 1 and self._interaction is not None and not is_interaction(
                self._interaction):
            if self._logger is not None:
                self._logger.error(
                    "'Interaction' instance expected, not '{}'".format(type(self._interaction)))
            raise TypeError(
                "'Interaction' instance expected, not '{}'".format(type(self._interaction)))

    def __str__(self):
        particles = '{} particle'.format(self._num_particles)

        if self._num_particles > 1:
            if self._interaction:
                particles = '{} interacting particles by {}'.format(
                    self._num_particles, self._interaction.to_string())
            else:
                particles = '{} particles'.format(self._num_particles)

        return '{} with {} and a {} over a {}'.format(
            'Discrete Time Quantum Walk', particles, self._coin.to_string(), self._mesh.to_string())

    def to_string(self):
        """Build a string representing this walk.

        Returns
        -------
        str
            The string representation of this walk.

        """
        return self.__str__()

    def _profile_operator(self, operator_type, operator, initial_time):
        if self._profiler is not None:
            app_id = self._spark_context.applicationId

            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_operator(
                '{}Operator'.format(operator_type),
                operator,
                (datetime.now() - initial_time).total_seconds())

            if self._logger is not None:
                self._logger.info(
                    "{} operator was built in {}s".format(
                        operator_type, info['buildingTime']))
                self._logger.info(
                    "{} operator is consuming {} bytes in memory and {} bytes in disk".format(
                        operator_type, info['memoryUsed'], info['diskUsed']))

            if Utils.get_conf(self._spark_context,
                              'quantum.dtqw.profiler.logExecutors') == 'True':
                self._profiler.log_executors(app_id=app_id)

    def _create_coin_operator(self, coord_format, storage_level):
        if self._logger is not None:
            self._logger.info("building coin operator...")

        initial_time = datetime.now()

        self._coin_operator = self._coin.create_operator(
            self._mesh, coord_format).materialize(storage_level)

        self._profile_operator(
            'coin',
            self._coin_operator,
            initial_time)

    def _create_shift_operator(self, coord_format, storage_level):
        if self._logger is not None:
            self._logger.info("building shift operator...")

        initial_time = datetime.now()

        self._shift_operator = self._mesh.create_operator(
            coord_format).materialize(storage_level)

        self._profile_operator(
            'shift',
            self._shift_operator,
            initial_time)

    def _create_interaction_operator(self, coord_format, storage_level):
        if self._logger is not None:
            self._logger.info("building interaction operator...")

        initial_time = datetime.now()

        self._interaction_operator = self._interaction.create_operator(
            coord_format).materialize(storage_level)

        self._profile_operator(
            'interaction',
            self._interaction_operator,
            initial_time)

    def _create_walk_operators(self, coord_format, storage_level):
        """Build the walk operator for the walk.

        When performing a multiparticle walk, this method builds a list with n operators,
        where n is the number of particles of the system. In this case, each operator is built by
        applying a tensor product between the evolution operator and ``n-1`` identity matrices as follows:

            ``W1 = W1 (X) I2 (X) ... (X) In
            Wi = I1 (X) ... (X) Ii-1 (X) Wi (X) Ii+1 (X) ... In
            Wn = I1 (X) ... (X) In-1 (X) Wn``

        Regardless the number of particles, the walk operators have their ``(i,j,value)`` coordinates converted to
        appropriate coordinates for multiplication, in this case, the :py:const:`sparkquantum.utils.Utils.MatrixCoordinateMultiplier`.

        Parameters
        ----------
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault`.
        storage_level : :py:class:`pyspark.StorageLevel`, optional
            The desired storage level when materializing the RDD.
            Default value is :py:const:`pyspark.StorageLevel.MEMORY_AND_DISK`.

        Raises
        ------
        ValueError
            If the chosen 'quantum.dtqw.walkOperator.kroneckerMode' configuration is not valid.

        """
        app_id = self._spark_context.applicationId

        if self._logger is not None:
            self._logger.info("building walk operator...")

        if self._coin_operator is None:
            if self._logger is not None:
                self._logger.info(
                    "no coin operator has been set. A new one will be built")

            self._create_coin_operator(
                Utils.MatrixCoordinateMultiplicand, storage_level)

        if self._shift_operator is None:
            if self._logger is not None:
                self._logger.info(
                    "no shift operator has been set. A new one will be built")

            self._create_shift_operator(
                Utils.MatrixCoordinateMultiplier, storage_level)

        if self._num_particles == 1:
            initial_time = datetime.now()

            evolution_operator = self._shift_operator.multiply(
                self._coin_operator, coord_format).persist(storage_level)

            if Utils.get_conf(self._spark_context,
                              'quantum.dtqw.walkOperator.checkpoint') == 'True':
                evolution_operator = evolution_operator.checkpoint()

            self._walk_operators = (
                evolution_operator.materialize(storage_level), )

            self._coin_operator.unpersist()
            self._shift_operator.unpersist()

            self._profile_operator(
                'walk', self._walk_operators[0], initial_time)
        else:
            t_tmp = datetime.now()

            evolution_operator = self._shift_operator.multiply(
                self._coin_operator, Utils.MatrixCoordinateDefault
            ).materialize(storage_level)

            self._coin_operator.unpersist()
            self._shift_operator.unpersist()

            shape = evolution_operator.shape
            shape_tmp = shape

            self._walk_operators = []

            kron_mode = Utils.get_conf(
                self._spark_context, 'quantum.dtqw.walkOperator.kroneckerMode')

            if kron_mode != Utils.KroneckerModeBroadcast and kron_mode != Utils.KroneckerModeDump:
                if self._logger is not None:
                    self._logger.error("invalid kronecker mode")
                raise ValueError("invalid kronecker mode")

            if kron_mode == Utils.KroneckerModeBroadcast:
                eo = Utils.broadcast(self._spark_context,
                                     evolution_operator.data.collect())

                evolution_operator.unpersist()
            elif kron_mode == Utils.KroneckerModeDump:
                path = Utils.get_temp_path(
                    Utils.get_conf(self._spark_context,
                                   'quantum.dtqw.walkOperator.tempPath')
                )

                evolution_operator.dump(path)

            for p in range(self._num_particles):
                if self._logger is not None:
                    self._logger.debug(
                        "building walk operator for particle {}...".format(p + 1))

                # shape = shape_tmp

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
                    t_tmp = datetime.now()

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

                if coord_format == Utils.MatrixCoordinateMultiplier or coord_format == Utils.MatrixCoordinateMultiplicand:
                    rdd = Utils.change_coordinate(
                        rdd, Utils.MatrixCoordinateDefault, new_coord=coord_format
                    )

                    expected_elems = evolution_operator.num_nonzero_elements * \
                        evolution_operator.shape[0] ** (
                            self._num_particles - 1)
                    expected_size = Utils.get_size_of_type(
                        complex) * expected_elems
                    num_partitions = Utils.get_num_partitions(
                        self._spark_context, expected_size)

                    if num_partitions:
                        rdd = rdd.partitionBy(
                            numPartitions=num_partitions
                        )

                wo = Operator(
                    rdd, shape, coord_format
                ).persist(storage_level)

                if Utils.get_conf(
                        self._spark_context, 'quantum.dtqw.walkOperator.checkpoint') == 'True':
                    wo = wo.checkpoint()

                self._walk_operators.append(wo.materialize(storage_level))

                if self._profiler is not None:
                    self._profiler.profile_resources(app_id)
                    self._profiler.profile_executors(app_id)

                    info = self._profiler.profile_operator(
                        'walkOperatorParticle{}'.format(p + 1),
                        self._walk_operators[-1], (datetime.now() -
                                                   t_tmp).total_seconds()
                    )

                    if self._logger is not None:
                        self._logger.info(
                            "walk operator for particle {} was built in {}s".format(
                                p + 1, info['buildingTime'])
                        )
                        self._logger.info(
                            "walk operator for particle {} is consuming {} bytes in memory and {} bytes in disk".format(
                                p + 1, info['memoryUsed'], info['diskUsed']
                            )
                        )

                    if Utils.get_conf(
                            self._spark_context, 'quantum.dtqw.profiler.logExecutors') == 'True':
                        self._profiler.log_executors(app_id=app_id)

            if kron_mode == Utils.KroneckerModeBroadcast:
                eo.unpersist()
            elif kron_mode == Utils.KroneckerModeDump:
                evolution_operator.unpersist()

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
        if self._logger is not None:
            self._logger.info("destroying operators...")

        self._destroy_coin_operator()
        self._destroy_shift_operator()
        self._destroy_interaction_operator()
        self._destroy_walk_operators()

        if self._logger is not None:
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

    def walk(self, steps, initial_state,
             storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform a walk.

        Parameters
        ----------
        steps : int
            The number of steps of the walk.
        initial_state : :py:class:`sparkquantum.dtqw.state.State`
            The initial state of the system.
        storage_level : :py:class:`pyspark.StorageLevel`, optional
            The desired storage level when materializing the RDD.
            Default value is :py:const:`pyspark.StorageLevel.MEMORY_AND_DISK`.

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
            if self._logger is not None:
                self._logger.error("the number of steps must be positive")
            raise ValueError("the number of steps must be positive")

        if not self._mesh.check_steps(steps):
            if self._logger is not None:
                self._logger.error(
                    "invalid number of steps for the chosen mesh")
            raise ValueError("invalid number of steps for the chosen mesh")

        if self._logger is not None:
            self._logger.info("steps: {}".format(steps))
            self._logger.info("mesh: {}".format(self._mesh.to_string()))
            self._logger.info(
                "number of particles: {}".format(self._num_particles))

            if self._num_particles > 1:
                if self._interaction is None:
                    self._logger.info(
                        "no interaction between particles has been defined")
                else:
                    self._logger.info(
                        "interaction between particles: {}".format(
                            self._interaction.to_string()))

            if self._mesh.broken_links is None:
                self._logger.info("no broken links have been defined")
            else:
                self._logger.info("broken links probability: {}".format(
                    self._mesh.broken_links.to_string()))

        result = initial_state.materialize(storage_level)

        app_id = self._spark_context.applicationId

        if self._profiler is not None:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_state('initialState', result, 0.0)

            if self._logger is not None:
                self._logger.info(
                    "initial state is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

        if self._logger is not None:
            self._profiler.log_rdd(app_id=app_id)

        configs = self._get_walk_configs()

        t1 = datetime.now()

        if self._logger is not None:
            self._logger.info("starting the walk...")

        # Building walk operators once if not simulating decoherence with
        # random broken links
        if not self._mesh.broken_links or not self._mesh.broken_links.is_random():
            self._create_walk_operators(
                Utils.MatrixCoordinateMultiplier, storage_level)

        if self._num_particles > 1 and self._interaction is not None and self._interaction_operator is None:
            self._create_interaction_operator(
                Utils.MatrixCoordinateMultiplier, storage_level)

        for i in range(1, steps + 1, 1):
            # When there is a broken links probability, the walk operators will
            # be built in each step of the walk
            if self._mesh.broken_links and self._mesh.broken_links.is_random():
                self._destroy_shift_operator()
                self._destroy_walk_operators()
                self._create_walk_operators(
                    Utils.MatrixCoordinateMultiplier, storage_level)

            t_tmp = datetime.now()

            result_tmp = result

            if self._num_particles == 1:
                result_tmp = self._walk_operators[0].multiply(result_tmp)
            else:
                if self._interaction_operator is not None:
                    result_tmp = self._interaction_operator.multiply(
                        result_tmp)

                for wo in reversed(self._walk_operators):
                    result_tmp = wo.multiply(result_tmp)

            # In the last step, the resulting state is not materialized
            # because it will be repartitioned to a more appropriate
            # number of partitions and have a partitioner defined.
            if i == steps:
                expected_elems = result_tmp.shape[0]
                expected_size = Utils.get_size_of_type(
                    result_tmp.data_type) * expected_elems
                num_partitions = Utils.get_num_partitions(
                    self._spark_context, expected_size)

                if num_partitions:
                    result_tmp.define_partitioner(num_partitions)

            result_tmp.materialize(storage_level)
            result.unpersist()

            if configs['checkpointing_frequency'] >= 0 and i % configs['checkpointing_frequency'] == 0:
                result_tmp.checkpoint()

            if configs['dumping_frequency'] >= 0 and i % configs['dumping_frequency'] == 0:
                result_tmp.dump(
                    configs['dumping_path'] + "states/" + str(i))

            if configs['check_unitary'] == 'True':
                if not result_tmp.is_unitary():
                    if self._logger is not None:
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

            result = result_tmp

            if self._profiler is not None:
                self._profiler.profile_resources(app_id)
                self._profiler.profile_executors(app_id)

                info = self._profiler.profile_state(
                    'systemState{}'.format(
                        i), result, (datetime.now() - t_tmp).total_seconds()
                )

                if self._logger is not None:
                    self._logger.info(
                        "step was done in {}s".format(info['buildingTime']))
                    self._logger.info(
                        "system state of step {} is consuming {} bytes in memory and {} bytes in disk".format(
                            i, info['memoryUsed'], info['diskUsed']
                        )
                    )

            if self._logger is not None:
                self._profiler.log_rdd(app_id=app_id)

        if self._logger is not None:
            self._logger.info("walk was done in {}s".format(
                (datetime.now() - t1).total_seconds()))

        t1 = datetime.now()

        if self._logger is not None:
            self._logger.debug("checking if the final state is unitary...")

        if not result.is_unitary():
            if self._logger is not None:
                self._logger.error("the final state is not unitary")
            raise ValueError("the final state is not unitary")

        if self._logger is not None:
            self._logger.debug("unitarity check was done in {}s".format(
                (datetime.now() - t1).total_seconds()))

        if self._profiler is not None:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_state('finalState', result, 0.0)

            if self._logger is not None:
                self._logger.info(
                    "final state is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

            if Utils.get_conf(self._spark_context,
                              'quantum.dtqw.profiler.logExecutors') == 'True':
                self._profiler.log_executors(app_id=app_id)

        return result
