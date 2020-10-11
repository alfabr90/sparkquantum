from pyspark import SparkContext, StorageLevel

from sparkquantum import conf, constants, util
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.dtqw.particle import is_particle
from sparkquantum.math import util as mathutil
from sparkquantum.math.matrix import Matrix, is_matrix

__all__ = ['State', 'is_state']


class State(Matrix):
    """Class for the system state."""

    def __init__(self, rdd, shape, mesh, particles,
                 dtype=complex, coord_format=constants.MatrixCoordinateDefault,
                 repr_format=constants.StateRepresentationFormatCoinPosition, nelem=None):
        """Build a state object.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this object. Must be 2-dimensional.
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles are walking on.
        particles : tuple of :py:class:`sparkquantum.dtqw.particle.Particle`
            The particles present in the walk.
        dtype : type, optional
            The Python type of all values in this object. Default value is complex.
        coord_format : int, optional
            The coordinate format of this object. Default value is :py:const:`sparkquantum.constants.MatrixCoordinateDefault`.
        repr_format : int, optional
            Indicate how the quantum system is represented.
            Default value is :py:const:`sparkquantum.constants.StateRepresentationFormatCoinPosition`.
        nelem : int, optional
            The expected (or definitive) number of elements. This helps to find a
            better number of partitions when (re)partitioning the RDD. Default value is None.

        """
        if not is_mesh(mesh):
            raise TypeError(
                "'Mesh' instance expected, not '{}'".format(type(mesh)))

        if len(particles) < 1:
            raise ValueError("invalid number of particles")

        for p in particles:
            if not is_particle(p):
                raise TypeError(
                    "'Particle' instance expected, not '{}'".format(type(p)))

        super().__init__(rdd, shape,
                         dtype=dtype, coord_format=coord_format, nelem=nelem)

        self._mesh = mesh
        self._particles = particles

        self._repr_format = repr_format

    @ property
    def mesh(self):
        """:py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`"""
        return self._mesh

    @ property
    def particles(self):
        """int"""
        return self._particles

    @ property
    def repr_format(self):
        """int"""
        return self._repr_format

    def __str__(self):
        if self._particles == 1:
            particles = '1 particle'
        else:
            particles = '{} particles'.format(self._particles)

        return 'State with shape {} of {} over a {}'.format(
            self._shape, particles, self._mesh)

    def dump(self, mode, glue=' ', path=None, codec=None,
             filename=None, format=constants.StateDumpingFormatIndex):
        """Dump this object's RDD to disk in a unique file or in many part-* files.

        Notes
        -----
        Depending on the chosen dumping mode, this method calls the :py:func:`pyspark.RDD.collect` method.
        This is not suitable for large working sets, as all data may not fit into driver's main memory.

        Parameters
        ----------
        mode : int
            Storage mode used to dump this state.
        glue : str, optional
            The glue string that connects each component of each element in the RDD.
            Default value is ' '.
        codec : str, optional
            Codec name used to compress the dumped data. Default value is None.
        filename : str, optional
            The full path with file name used when the dumping mode is in a single file.
            Default value is None.
        format : int, optional
            Printing format used to dump this state. Default value is :py:const:`sparkquantum.constants.StateDumpingFormatIndex`.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If this state's coordinate format is not :py:const:`sparkquantum.constants.MatrixCoordinateDefault` or
            if the chosen dumping mode or dumping format is not valid.

        """
        if self._coord_format != constants.MatrixCoordinateDefault:
            self._logger.error("invalid coordinate format")
            raise ValueError("invalid coordinate format")

        rdd = self.clear().data

        if format == constants.StateDumpingFormatIndex:
            rdd = rdd.map(
                lambda m: glue.join((str(m[0]), str(m[1]), str(m[2])))
            )
        elif format == constants.StateDumpingFormatCoordinate:
            repr_format = self._repr_format

            ndim = self._mesh.ndim
            csubspace = 2
            cspace = csubspace ** ndim
            psubspace = self._mesh.shape
            pspace = self._mesh.sites
            particles = self._particles
            cpspace = cspace * pspace

            if ndim == 1:
                mesh_offset = min(self._mesh.axis()[0])

                if repr_format == constants.StateRepresentationFormatCoinPosition:
                    def __map(m):
                        ix = []

                        for p in range(particles):
                            # Coin
                            ix.append(
                                str(int(m[0] / (cpspace ** (particles - 1 - p) * psubspace[0])) % csubspace))
                            # Position
                            ix.append(
                                str(int(m[0] / (cpspace ** (particles - 1 - p))) % psubspace[0] + mesh_offset))

                        ix.append(str(m[2]))

                        return glue.join(ix)
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(m):
                        xi = []

                        for p in range(particles):
                            # Position
                            xi.append(str(int(
                                m[0] / (cpspace ** (particles - 1 - p) * csubspace)) % psubspace[0] + mesh_offset))
                            # Coin
                            xi.append(
                                str(int(m[0] / (cpspace ** (particles - 1 - p))) % csubspace))

                        xi.append(str(m[2]))

                        return glue.join(xi)
                else:
                    self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")
            elif ndim == 2:
                axis = self._mesh.axis()
                mesh_offset = (min(axis[0]), min(axis[1]))

                if repr_format == constants.StateRepresentationFormatCoinPosition:
                    def __map(m):
                        ijxy = []

                        for p in range(particles):
                            # Coin
                            ijxy.append(str(int(
                                m[0] / (cpspace ** (particles - 1 - p) * csubspace * psubspace[0] * psubspace[1])) % csubspace))
                            ijxy.append(str(int(
                                m[0] / (cpspace ** (particles - 1 - p) * psubspace[0] * psubspace[1])) % csubspace))
                            # Position
                            ijxy.append(str(int(
                                m[0] / (cpspace ** (particles - 1 - p) * psubspace[1])) % psubspace[0] + mesh_offset[0]))
                            ijxy.append(
                                str(int(m[0] / (cpspace ** (particles - 1 - p))) % psubspace[1] + mesh_offset[1]))

                        ijxy.append(str(m[2]))

                        return glue.join(ijxy)
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(m):
                        xyij = []

                        for p in range(particles):
                            # Position
                            xyij.append(str(int(
                                m[0] / (cpspace ** (particles - 1 - p) * cspace * psubspace[1])) % psubspace[0] + mesh_offset[0]))
                            xyij.append(str(int(
                                m[0] / (cpspace ** (particles - 1 - p) * cspace)) % psubspace[1] + mesh_offset[1]))
                            # Coin
                            xyij.append(str(int(
                                m[0] / (cpspace ** (particles - 1 - p) * csubspace)) % csubspace))
                            xyij.append(
                                str(int(m[0] / (cpspace ** (particles - 1 - p))) % csubspace))

                        xyij.append(str(m[2]))

                        return glue.join(xyij)
                else:
                    self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")
            else:
                self._logger.error("mesh dimension not implemented")
                raise NotImplementedError("mesh dimension not implemented")

            rdd = rdd.map(__map)
        else:
            self._logger.error("invalid dumping format")
            raise ValueError("invalid dumping format")

        if mode == constants.DumpingModeUniqueFile:
            data = rdd.collect()

            with open(filename, 'a') as f:
                for d in data:
                    f.write(d + "\n")
        elif mode == constants.DumpingModePartFiles:
            rdd.saveAsTextFile(path, codec)
        else:
            self._logger.error("invalid dumping mode")
            raise ValueError("invalid dumping mode")

    def to_coordinate(self, coord_format):
        """Change the coordinate format of this object.

        Notes
        -----
        Due to the immutability of RDD, a new RDD instance is created
        in the desired coordinate format. Thus, a new instance of this class
        is returned with this RDD.

        Parameters
        ----------
        coord_format : int
            The new coordinate format of this object.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.state.State`
            A new state object with the RDD in the desired coordinate format.

        """
        return State.from_matrix(super().to_coordinate(coord_format),
                                 self._mesh, self._particles, self._repr_format)

    def transpose(self):
        """Transpose this state.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.state.State`
            The resulting state.

        """
        return State.from_matrix(super().transpose(),
                                 self._mesh, self._particles, self._repr_format)

    def kron(self, other):
        """Perform a tensor (Kronecker) product with another system state.

        Parameters
        ----------
        other : :py:class:`sparkquantum.dtqw.state.State`
            The other system state.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.state.State`
            The resulting state.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.dtqw.state.State`.

        """
        if not is_state(other):
            self._logger.error(
                "'State' instance expected, not '{}'".format(type(other)))
            raise TypeError(
                "'State' instance expected, not '{}'".format(type(other)))

        return State.from_matrix(super().kron(other),
                                 self._mesh, self._particles + other.particles, self._repr_format)

    def clear(self):
        return None

    def copy(self):
        return None

    def sum(self, other):
        return None

    def subtract(self, other):
        return None

    def multiply(self, other):
        return None

    def divide(self, other):
        return None

    def dot_product(self, other):
        return None

    @staticmethod
    def from_matrix(matrix, mesh, particles, repr_format):
        """Build a state from a matrix object.

        Parameters
        ----------
        matrix : :py:class:`sparkquantum.math.matrix.Matrix`
            The matrix to serve as a base.
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles are walking on.
        particles : tuple of :py:class:`sparkquantum.dtqw.particle.Particle`
            The particles present in the walk.
        repr_format : int, optional
            Indicate how the quantum system is represented.
            Default value is :py:const:`sparkquantum.constants.StateRepresentationFormatCoinPosition`.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.state.State`
            The new state.

        Raises
        ------
        TypeError
            If `other` is not a :py:class:`sparkquantum.math.matrix.Matrix`.

        """
        if not is_matrix(matrix):
            raise TypeError(
                "'Matrix' instance expected, not '{}'".format(type(matrix)))

        return State(matrix.data, matrix.shape, mesh, particles,
                     dtype=matrix.dtype, coord_format=matrix.coord_format, repr_format=repr_format, nelem=matrix.nelem)


def is_state(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.state.State` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.state.State` object, False otherwise.

    """
    return isinstance(obj, State)
