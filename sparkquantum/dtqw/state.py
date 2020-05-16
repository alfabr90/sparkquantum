from datetime import datetime
import math

from pyspark import SparkContext, StorageLevel

from sparkquantum.dtqw.coin.coin import is_coin
from sparkquantum.dtqw.interaction.interaction import is_interaction
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.math.vector import Vector
from sparkquantum.utils.utils import Utils

__all__ = ['State', 'is_state']


class State(Vector):
    """Class for the system state."""

    def __init__(self, rdd, shape, coin, mesh,
                 num_particles, interaction=None):
        """Build a state object.

        Parameters
        ----------
        rdd : :py:class:`pyspark.RDD`
            The base RDD of this object.
        shape : tuple
            The shape of this state object. Must be a two-dimensional tuple.
        coin : :py:class:`sparkquantum.dtqw.coin.coin.Coin`
            The coin for the walk.
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles are walking on.
        num_particles : int
            The number of particles present in the walk.
        interaction : :py:class:`sparkquantum.dtqw.interaction.interaction.Interaction`, optional
            A particles interaction object.

        """
        super().__init__(rdd, shape, data_type=complex)

        self._coin = coin
        self._mesh = mesh
        self._num_particles = num_particles
        self._interaction = interaction

        if not is_coin(self._coin):
            self._logger.error(
                "'Coin' instance expected, not '{}'".format(type(self._coin)))
            raise TypeError(
                "'Coin' instance expected, not '{}'".format(type(self._coin)))

        if not is_mesh(mesh):
            self._logger.error(
                "'Mesh' instance expected, not '{}'".format(
                    type(mesh)))
            raise TypeError(
                "'Mesh' instance expected, not '{}'".format(
                    type(mesh)))

        if self._num_particles is None or self._num_particles < 1:
            self._logger.error(
                "invalid number of particles. It must be greater than or equal to 1")
            raise ValueError(
                "invalid number of particles. It must be greater than or equal to 1")

        if self._num_particles > 1 and self._interaction is not None and not is_interaction(
                self._interaction):
            self._logger.error(
                "'Interaction' instance expected, not '{}'".format(type(self._interaction)))
            raise TypeError(
                "'Interaction' instance expected, not '{}'".format(type(self._interaction)))

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

    def __del__(self):
        # In cases where multiple simulations are performed,
        # the same Python's logger object is used for the same class name.
        # Removing all its handlers, the object is reset.
        for h in self._logger.handlers:
            self._logger.removeHandler(h)

    def __str__(self):
        if self._num_particles == 1:
            particles = 'one particle'
        else:
            particles = '{} particles'.format(self._num_particles)

        return 'Quantum State with shape {} of {} over a {}'.format(
            self._shape, particles, self._mesh)

    def dump(self, path, glue=None, codec=None,
             filename='', dumping_format=None):
        """Dump this object's RDD to disk in a unique file or in many part-* files.

        Notes
        -----
        This method checks the dumping format by using the 'quantum.dtqw.state.dumpingFormat' configuration value.
        In case the chosen format is the mesh coordinates one, this method also checks the state's representation format.
        Depending on the chosen dumping mode, this method calls the :py:func:`pyspark.RDD.collect` method.
        This is not suitable for large working sets, as all data may not fit into driver's main memory.

        Parameters
        ----------
        path : str
            The path where the dumped RDD will be located at.
        glue : str, optional
            The glue string that connects each coordinate and value of each element in the RDD.
            Default value is None. In this case, it uses the 'quantum.dumpingGlue' configuration value.
        codec : str, optional
            Codec name used to compress the dumped data.
            Default value is None. In this case, it uses the 'quantum.dumpingCompressionCodec' configuration value.
        filename : str, optional
            File name used when the dumping mode is in a single file. Default value is None.
            In this case, a temporary named file is generated inside the informed path.
        dumping_format : int, optional
            Printing format used to dump this state.
            Default value is None. In this case, it uses the 'quantum.math.dumpingFormat' configuration value.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            If any of the chosen 'quantum.dtqw.state.dumpingFormat', 'quantum.math.dumpingMode' or
            'quantum.dtqw.state.representationFormat' configuration is not valid.

        """
        if glue is None:
            glue = Utils.get_conf(self._spark_context, 'quantum.dumpingGlue')

        if codec is None:
            codec = Utils.get_conf(self._spark_context,
                                   'quantum.dumpingCompressionCodec')

        if dumping_format is None:
            dumping_format = int(Utils.get_conf(
                self._spark_context, 'quantum.dtqw.state.dumpingFormat'))

        dumping_mode = int(Utils.get_conf(
            self._spark_context, 'quantum.math.dumpingMode'))

        if dumping_format == Utils.StateDumpingFormatIndex:
            if dumping_mode == Utils.DumpingModeUniqueFile:
                data = self.data.collect()

                Utils.create_dir(path)

                if not filename:
                    filename = Utils.get_temp_path(path)
                else:
                    filename = Utils.append_slash_dir(path) + filename

                if len(data):
                    with open(filename, 'a') as f:
                        for d in data:
                            f.write(glue.join([str(e) for e in d]) + "\n")
            elif dumping_mode == Utils.DumpingModePartFiles:
                self.data.map(
                    lambda m: glue.join([str(e) for e in m])
                ).saveAsTextFile(path, codec)
            else:
                self._logger.error("invalid dumping mode")
                raise ValueError("invalid dumping mode")
        elif dumping_format == Utils.StateDumpingFormatCoordinate:
            repr_format = int(Utils.get_conf(
                self._spark_context, 'quantum.dtqw.state.representationFormat'))

            if self._mesh.dimension == 1:
                ndim = self._mesh.dimension
                coin_size = self._mesh.coin_size
                size = self._mesh.size
                num_particles = self._num_particles
                size_per_coin = int(coin_size / ndim)
                cs_size = size_per_coin * size

                mesh_offset = min(self._mesh.axis())

                if repr_format == Utils.StateRepresentationFormatCoinPosition:
                    def __map(m):
                        ix = []

                        for p in range(num_particles):
                            # Coin
                            ix.append(
                                str(int(m[0] / (cs_size ** (num_particles - 1 - p) * size)) % size_per_coin))
                            # Position
                            ix.append(
                                str(int(m[0] / (cs_size ** (num_particles - 1 - p))) % size + mesh_offset))

                        ix.append(str(m[1]))

                        return glue.join(ix)
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                    def __map(m):
                        xi = []

                        for p in range(num_particles):
                            # Position
                            xi.append(str(int(
                                m[0] / (cs_size ** (num_particles - 1 - p) * size_per_coin)) % size + mesh_offset))
                            # Coin
                            xi.append(
                                str(int(m[0] / (cs_size ** (num_particles - 1 - p))) % size_per_coin))

                        xi.append(str(m[1]))

                        return glue.join(xi)
                else:
                    self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")
            elif self._mesh.dimension == 2:
                ndim = self._mesh.dimension
                coin_size = self._mesh.coin_size
                size_x, size_y = self._mesh.size
                num_particles = self._num_particles
                size_per_coin = int(coin_size / ndim)
                cs_size_x = size_per_coin * size_x
                cs_size_y = size_per_coin * size_y
                cs_size_xy = cs_size_x * cs_size_y

                axis = self._mesh.axis()
                mesh_offset_x, mesh_offset_y = axis[0][0][0], axis[1][0][0]

                if repr_format == Utils.StateRepresentationFormatCoinPosition:
                    def __map(m):
                        ijxy = []

                        for p in range(num_particles):
                            # Coin
                            ijxy.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * cs_size_x * size_y)) % size_per_coin))
                            ijxy.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * size_x * size_y)) % size_per_coin))
                            # Position
                            ijxy.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * size_y)) % size_x + mesh_offset_x))
                            ijxy.append(
                                str(int(m[0] / (cs_size_xy ** (num_particles - 1 - p))) % size_y + mesh_offset_y))

                        ijxy.append(str(m[1]))

                        return glue.join(ijxy)
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                    def __map(m):
                        xyij = []

                        for p in range(num_particles):
                            # Position
                            xyij.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * coin_size * size_y)) % size_x + mesh_offset_x))
                            xyij.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * coin_size)) % size_y + mesh_offset_y))
                            # Coin
                            xyij.append(str(int(
                                m[0] / (cs_size_xy ** (num_particles - 1 - p) * size_per_coin)) % size_per_coin))
                            xyij.append(
                                str(int(m[0] / (cs_size_xy ** (num_particles - 1 - p))) % size_per_coin))

                        xyij.append(str(m[1]))

                        return glue.join(xyij)
                else:
                    self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")
            else:
                self._logger.error("mesh dimension not implemented")
                raise NotImplementedError("mesh dimension not implemented")

            if dumping_mode == Utils.DumpingModeUniqueFile:
                data = self.data.collect()

                Utils.create_dir(path)

                if not filename:
                    filename = Utils.get_temp_path(path)
                else:
                    filename = Utils.append_slash_dir(path) + filename

                if len(data):
                    with open(filename, 'a') as f:
                        for d in data:
                            f.write(glue.join(d) + "\n")
            elif dumping_mode == Utils.DumpingModePartFiles:
                self.data.map(
                    __map
                ).saveAsTextFile(path, codec)
            else:
                self._logger.error("invalid dumping mode")
                raise ValueError("invalid dumping mode")
        else:
            self._logger.error("invalid dumping format")
            raise ValueError("invalid dumping format")

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

        rdd, new_shape = self._kron(other)

        return State(rdd, new_shape, self._mesh, self._num_particles)

    @staticmethod
    def create(coin, mesh, positions, amplitudes, interaction=None,
               representationFormat=Utils.StateRepresentationFormatCoinPosition):
        """Create a system state.

        For system states with entangled particles, the state must be created
        by its class construct method.

        Parameters
        ----------
        coin : :py:class:`sparkquantum.dtqw.coin.coin.Coin`
            The coin for the walk.
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles are walking on.
        positions : tuple or list
            The position of each particle present in the quantum walk.
        amplitudes : tuple or list
            The amplitudes for each qubit of each particle in the quantum walk.
        interaction : :py:class:`sparkquantum.dtqw.interaction.interaction.Interaction`, optional
            A particles interaction object.
        representationFormat : int, optional
            Indicate how the quantum system will be represented.
            Default value is :py:const:`sparkquantum.utils.Utils.StateRepresentationFormatCoinPosition`.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.state.State`
            A new system state.

        Raises
        ------
        NotImplementedError
            If the dimension of the mesh is not valid.

        ValueError
            Iif the length of `positions` and `amplitudes` are not compatible (or invalid) or
            if the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid.

        TypeError
            If `coin` is not a :py:class:`sparkquantum.dtqw.coin.coin.Coin`,
            if `mesh` is not a :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh` or
            if `interaction` is not a :py:class:`sparkquantum.dtqw.interaction.interaction.Interaction`.

        """
        spark_context = SparkContext.getOrCreate()

        logger = Utils.get_logger(spark_context, State.__name__)

        if not is_coin(coin):
            logger.error(
                "'Coin' instance expected, not '{}'".format(type(coin)))
            raise TypeError(
                "'Coin' instance expected, not '{}'".format(type(coin)))

        if not is_mesh(mesh):
            logger.error(
                "'Mesh' instance expected, not '{}'".format(type(mesh)))
            raise TypeError(
                "'Mesh' instance expected, not '{}'".format(type(mesh)))

        coin_size = coin.size

        if mesh.dimension == 1:
            mesh_size = mesh.size
        elif mesh.dimension == 2:
            mesh_size = mesh.size[0] * mesh.size[1]
        else:
            logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        base_states = []

        num_particles = len(positions)

        if num_particles < 1:
            logger.error(
                "invalid number of particles. It must be greater than or equal to 1")
            raise ValueError(
                "invalid number of particles. It must be greater than or equal to 1")

        if num_particles != len(amplitudes):
            logger.error(
                "incompatible length of positions ({}) and amplitudes ({})".format(num_particles, len(amplitudes)))
            raise ValueError(
                "incompatible length of positions ({}) and amplitudes ({})".format(num_particles, len(amplitudes)))

        if num_particles > 1 and interaction is not None and not is_interaction(
                interaction):
            logger.error(
                "'Interaction' instance expected, not '{}'".format(type(interaction)))
            raise TypeError(
                "'Interaction' instance expected, not '{}'".format(type(interaction)))

        shape = (coin_size * mesh_size, 1)

        for p in range(num_particles):
            if representationFormat == Utils.StateRepresentationFormatCoinPosition:
                state = (
                    (a * mesh_size + positions[p],
                     amplitudes[p][a]) for a in range(
                        len(amplitudes[p])))
            elif representationFormat == Utils.StateRepresentationFormatPositionCoin:
                state = (
                    (positions[p] * coin_size + a,
                     amplitudes[p][a]) for a in range(
                        len(amplitudes[p])))
            else:
                logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            rdd = spark_context.parallelize(state)

            base_states.append(
                State(
                    rdd,
                    shape,
                    coin,
                    mesh,
                    num_particles,
                    interaction))

        initial_state = base_states[0]

        for p in range(1, num_particles):
            initial_state = initial_state.kron(base_states[p])

        for bs in base_states:
            bs.destroy()

        return initial_state


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
