import cmath

from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum.dtqw.interaction.interaction import Interaction
from sparkquantum.dtqw.operator import Operator
from sparkquantum.utils.utils import Utils

__all__ = ['CollisionPhaseInteraction']


class CollisionPhaseInteraction(Interaction):
    """Class that represents interaction between particles defined by
    a phase change during collisions."""

    def __init__(self, num_particles, mesh, collision_phase,
                 logger=None, profiler=None):
        """Build a interaction object defined by a phase change during collisions.

        Parameters
        ----------
        num_particles : int
            The number of particles present in the walk.
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles will walk over.
        collision_phase : complex
            The phase change applied during collisions.
        logger : py:class:`sparkquantum.utils.logger.Logger`, optional
            A logger object.
        profiler : py:class:`sparkquantum.utils.profiler.Profiler`, optional
            A profiler object.

        Raises
        ------
        ValueError
            If the collision phase or the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid.

        """
        super().__init__(num_particles, mesh, logger, profiler)

        if not collision_phase:
            if self._logger is not None:
                self._logger.error(
                    "no collision phase or a zeroed collision phase was informed")
            raise ValueError(
                "no collision phase or a zeroed collision phase was informed")

        self._collision_phase = collision_phase

    @property
    def collision_phase(self):
        """complex"""
        return self._collision_phase

    def __str__(self):
        return 'Collision Phase Interaction with phase value of {}'.format(
            self._num_particles)

    def _create_rdd(self, coord_format, storage_level):
        phase = cmath.exp(self._collision_phase * (0.0 + 1.0j))
        num_particles = self._num_particles

        repr_format = int(Utils.get_conf(self._spark_context,
                                         'quantum.dtqw.state.representationFormat'))

        if self._mesh.is_1d():
            ndim = self._mesh.dimension
            coin_size = self._mesh.coin_size
            size = self._mesh.size
            cs_size = int(coin_size / ndim) * size

            rdd_range = cs_size ** num_particles
            shape = (rdd_range, rdd_range)

            if repr_format == Utils.StateRepresentationFormatCoinPosition:
                def __map(m):
                    x = []

                    for p in range(num_particles):
                        x.append(
                            int(m / (cs_size ** (num_particles - 1 - p))) % size)

                    for p1 in range(num_particles):
                        for p2 in range(num_particles):
                            if p1 != p2 and x[p1] == x[p2]:
                                return m, m, phase

                    return m, m, 1
            elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                def __map(m):
                    x = []

                    for p in range(num_particles):
                        x.append(
                            int(m / (cs_size ** (num_particles - 1 - p) * coin_size)) % size)

                    for p1 in range(num_particles):
                        for p2 in range(num_particles):
                            if p1 != p2 and x[p1] == x[p2]:
                                return m, m, phase

                    return m, m, 1
            else:
                if self._logger is not None:
                    self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")
        elif self._mesh.is_2d():
            ndim = self._mesh.dimension
            coin_size = self._mesh.coin_size
            size_x, size_y = self._mesh.size
            cs_size_x = int(coin_size / ndim) * size_x
            cs_size_y = int(coin_size / ndim) * size_y
            cs_size_xy = cs_size_x * cs_size_y

            rdd_range = cs_size_xy ** num_particles
            shape = (rdd_range, rdd_range)

            if repr_format == Utils.StateRepresentationFormatCoinPosition:
                def __map(m):
                    xy = []

                    for p in range(num_particles):
                        xy.append(
                            (
                                int(m / (cs_size_xy **
                                         (num_particles - 1 - p) * size_y)) % size_x,
                                int(m / (cs_size_xy ** (num_particles - 1 - p))) % size_y
                            )
                        )

                    for p1 in range(num_particles):
                        for p2 in range(num_particles):
                            if p1 != p2 and xy[p1][0] == xy[p2][0] and xy[p1][1] == xy[p2][1]:
                                return m, m, phase

                    return m, m, 1
            elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                def __map(m):
                    xy = []

                    for p in range(num_particles):
                        xy.append(
                            (
                                int(m / (cs_size_xy ** (num_particles - 1 - p)
                                         * coin_size * size_y)) % size_x,
                                int(m / (cs_size_xy ** (num_particles -
                                                        1 - p) * coin_size)) % size_y
                            )
                        )

                    for p1 in range(num_particles):
                        for p2 in range(num_particles):
                            if p1 != p2 and xy[p1][0] == xy[p2][0] and xy[p1][1] == xy[p2][1]:
                                return m, m, phase

                    return m, m, 1
            else:
                if self._logger is not None:
                    self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")
        else:
            if self._logger is not None:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        rdd = self._spark_context.range(
            rdd_range
        ).map(
            __map
        )

        if coord_format == Utils.MatrixCoordinateMultiplier or coord_format == Utils.MatrixCoordinateMultiplicand:
            rdd = Utils.change_coordinate(
                rdd, Utils.MatrixCoordinateDefault, new_coord=coord_format
            )

            expected_elems = rdd_range
            expected_size = Utils.get_size_of_type(complex) * expected_elems
            num_partitions = Utils.get_num_partitions(
                self._spark_context, expected_size)

            if num_partitions:
                rdd = rdd.partitionBy(
                    numPartitions=num_partitions
                )

        return rdd, shape

    def create_operator(self, coord_format=Utils.MatrixCoordinateDefault,
                        storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build the interaction operator.

        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault`.
        storage_level : :py:class:`pyspark.StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is :py:const:`pyspark.StorageLevel.MEMORY_AND_DISK`.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid.

        """
        if self._logger is not None:
            self._logger.info("building interaction operator...")

        initial_time = datetime.now()

        rdd, shape = self._create_rdd(coord_format, storage_level)

        operator = Operator(
            rdd, shape, coord_format=coord_format).materialize(storage_level)

        self._profile(operator, initial_time)

        return operator
