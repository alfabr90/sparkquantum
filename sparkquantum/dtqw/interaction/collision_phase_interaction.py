import cmath
from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum import conf, constants, util
from sparkquantum.dtqw.interaction.interaction import Interaction
from sparkquantum.dtqw.operator import Operator

__all__ = ['CollisionPhaseInteraction']


class CollisionPhaseInteraction(Interaction):
    """Class that represents interaction between particles defined by
    a phase change during collisions."""

    def __init__(self, num_particles, mesh, collision_phase):
        """Build a interaction object defined by a phase change during collisions.

        Parameters
        ----------
        num_particles : int
            The number of particles present in the walk.
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles will walk over.
        collision_phase : complex
            The phase change applied during collisions.

        Raises
        ------
        ValueError
            If the collision phase or the chosen 'sparkquantum.dtqw.state.representationFormat'
            configuration is not valid.

        """
        super().__init__(num_particles, mesh)

        if not collision_phase:
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
            self._collision_phase)

    def create_operator(self):
        """Build the interaction operator.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If the chosen 'sparkquantum.dtqw.state.representationFormat' configuration is not valid.

        """
        phase = cmath.exp(self._collision_phase * (0.0 + 1.0j))
        num_particles = self._num_particles

        repr_format = int(conf.get_conf(self._spark_context,
                                        'sparkquantum.dtqw.state.representationFormat'))

        if self._mesh.dimension == 1:
            ndim = self._mesh.dimension
            coin_size = self._mesh.coin_size
            size = self._mesh.size
            cs_size = int(coin_size / ndim) * size

            rdd_range = cs_size ** num_particles
            shape = (rdd_range, rdd_range)

            num_elements = shape[0]

            if repr_format == constants.StateRepresentationFormatCoinPosition:
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
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
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
                self._logger.error(
                    "invalid representation format")
                raise ValueError("invalid representation format")
        elif self._mesh.dimension == 2:
            ndim = self._mesh.dimension
            coin_size = self._mesh.coin_size
            size_x, size_y = self._mesh.size
            cs_size_x = int(coin_size / ndim) * size_x
            cs_size_y = int(coin_size / ndim) * size_y
            cs_size_xy = cs_size_x * cs_size_y

            rdd_range = cs_size_xy ** num_particles
            shape = (rdd_range, rdd_range)

            num_elements = shape[0]

            if repr_format == constants.StateRepresentationFormatCoinPosition:
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
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
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
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")
        else:
            self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        rdd = self._spark_context.range(
            rdd_range
        ).map(
            __map
        )

        return Operator(rdd, shape, num_elements=num_elements)
