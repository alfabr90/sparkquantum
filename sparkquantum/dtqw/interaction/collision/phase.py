import cmath

from sparkquantum import constants
from sparkquantum.dtqw.interaction.collision.collision import Collision
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.dtqw.operator import Operator

__all__ = ['PhaseChange']


class PhaseChange(Collision):
    """Class that represents interaction between particles defined by
    a phase change during collisions."""

    def __init__(self, phase):
        """Build a interaction object defined by a phase change during collisions.

        Parameters
        ----------
        phase : complex
            The phase applied during collisions.

        """
        super().__init__()

        self._phase = phase

    @property
    def phase(self):
        """complex"""
        return self._phase

    def __str__(self):
        return 'Phase changing collision interaction with phase value of {}'.format(
            self._phase)

    def create_operator(self, mesh, particles,
                        repr_format=constants.StateRepresentationFormatCoinPosition):
        """Build the interaction operator for a quantum walk.

        Parameters
        ----------
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles are walking on.
        particles : int
            The number of particles present in the quantum walk.
        repr_format : int, optional
            Indicate how the quantum system is represented.
            Default value is :py:const:`sparkquantum.constants.StateRepresentationFormatCoinPosition`.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator`
            The created operator using this collision phase interaction.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If `repr_format` is not valid.

        """
        if not is_mesh(mesh):
            self._logger.error(
                "'Mesh' instance expected, not '{}'".format(type(mesh)))
            raise TypeError(
                "'Mesh' instance expected, not '{}'".format(type(mesh)))

        colval = cmath.exp(self._phase)

        ndim = mesh.ndim
        csubspace = 2
        cspace = csubspace ** ndim
        psubspace = mesh.shape
        pspace = mesh.sites
        cpspace = cspace * pspace

        shape = (cpspace ** particles, cpspace ** particles)

        dtype = int if complex(self._phase) == complex() else complex

        nelem = shape[0]

        if ndim == 1:
            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(m):
                    x = []

                    for p in range(particles):
                        x.append(
                            int(m / (cpspace ** (particles - 1 - p))) % pspace)

                    for p1 in range(particles):
                        for p2 in range(particles):
                            if p1 != p2 and x[p1] == x[p2]:
                                return m, m, colval

                    return m, m, 1
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(m):
                    x = []

                    for p in range(particles):
                        x.append(
                            int(m / (cpspace ** (particles - 1 - p) * cspace)) % pspace)

                    for p1 in range(particles):
                        for p2 in range(particles):
                            if p1 != p2 and x[p1] == x[p2]:
                                return m, m, colval

                    return m, m, 1
            else:
                self._logger.error(
                    "invalid representation format")
                raise ValueError("invalid representation format")
        elif ndim == 2:
            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(m):
                    xy = []

                    for p in range(particles):
                        xy.append(
                            (
                                int(m / (cpspace **
                                         (particles - 1 - p) * psubspace[1])) % psubspace[0],
                                int(m / (cpspace ** (particles - 1 - p))
                                    ) % psubspace[1]
                            )
                        )

                    for p1 in range(particles):
                        for p2 in range(particles):
                            if p1 != p2 and xy[p1][0] == xy[p2][0] and xy[p1][1] == xy[p2][1]:
                                return m, m, colval

                    return m, m, 1
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(m):
                    xy = []

                    for p in range(particles):
                        xy.append(
                            (
                                int(m / (cpspace ** (particles - 1 - p)
                                         * cspace * psubspace[1])) % psubspace[0],
                                int(m / (cpspace ** (particles -
                                                     1 - p) * cspace)) % psubspace[1]
                            )
                        )

                    for p1 in range(particles):
                        for p2 in range(particles):
                            if p1 != p2 and xy[p1][0] == xy[p2][0] and xy[p1][1] == xy[p2][1]:
                                return m, m, colval

                    return m, m, 1
            else:
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")
        else:
            self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        rdd = self._sc.range(
            shape[0]
        ).map(
            __map
        )

        return Operator(rdd, shape, dtype=dtype, nelem=nelem)
