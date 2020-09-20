from sparkquantum import constants, util
from sparkquantum.dtqw.mesh.grid.grid import Grid
from sparkquantum.dtqw.operator import Operator

__all__ = ['OneDimensional']


class OneDimensional(Grid):
    """Top-level class for one-dimensional grids."""

    def __init__(self, shape, percolation=None):
        """Build a top-level one-dimensional grid object.

        Parameters
        ----------
        shape : tuple of int
            Shape of this grid. Must be 1-dimensional.
        percolation : :py:class:`sparkquantum.dtqw.mesh.percolation.Percolation`, optional
            A percolation object.

        """
        super().__init__(shape, percolation=percolation)

        if len(shape) != 1:
            raise ValueError("invalid shape")

        # The number of edges is the same of the shape of the mesh.
        # For the already implemented types of mesh, the border edges are the same.
        #
        # An example of a 5x1 mesh:
        #
        # 00 O 01 O 02 O 03 O 04 O 00
        # ---------------------------
        #              x
        self._edges = self._shape[0]

    def __str__(self):
        """Build a string representing this grid.

        Returns
        -------
        str
            The string representation of this grid.

        """
        percolation = ''

        if self._percolation is not None:
            percolation = ' and {}'.format(self._percolation)

        return '1d grid with shape {}{}'.format(
            self._shape, percolation)

    def create_operator(self,
                        repr_format=constants.StateRepresentationFormatCoinPosition, perc_mode=constants.PercolationsGenerationModeBroadcast):
        """Build the shift operator for a quantum walk.

        Parameters
        ----------
        repr_format : int, optional
            Indicate how the quantum system is represented.
            Default value is :py:const:`sparkquantum.constants.StateRepresentationFormatCoinPosition`.
        perc_mode : int, optional
            Indicate how the percolations will be generated.
            Default value is :py:const:`sparkquantum.constants.PercolationsGenerationModeBroadcast`.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator`
            The created operator using this mesh.

        Raises
        ------
        ValueError
            If `repr_format` or `perc_mode` is not valid.

        """
        cspace = 2 ** self._ndim
        pspace = self._sites
        shape = (cspace * pspace, cspace * pspace)

        nelem = shape[0]

        if self._percolation is not None:
            percolations = self._percolation.generate(self._edges)

            if perc_mode == constants.PercolationsGenerationModeRDD:
                if repr_format == constants.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(cspace):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the
                            # vertex from the edge number
                            x = (e[1][0] - i - l) % pspace

                            if e[1][1]:
                                l = 0

                            yield (i + l) * pspace + (x + l) % pspace, (1 - i) * pspace + x, 1
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(cspace):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the
                            # vertex from the edge number
                            x = (e[1][0] - i - l) % pspace

                            if e[1][1]:
                                l = 0

                            yield ((x + l) % pspace) * cspace + i + l, x * cspace + 1 - i, 1
                else:
                    percolations.unpersist()

                    self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")

                rdd = self._sc.range(
                    self._edges
                ).map(
                    lambda m: (m, m)
                ).leftOuterJoin(
                    percolations
                ).flatMap(
                    __map
                )
            elif perc_mode == constants.PercolationsGenerationModeBroadcast:
                if repr_format == constants.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        for i in range(cspace):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the
                            # vertex from the edge number
                            x = (e - i - l) % pspace

                            if e in percolations.value:
                                l = 0

                            yield (i + l) * pspace + (x + l) % pspace, (1 - i) * pspace + x, 1
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        for i in range(cspace):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the
                            # vertex from the edge number
                            x = (e - i - l) % pspace

                            if e in percolations.value:
                                l = 0

                            yield ((x + l) % pspace) * cspace + i + l, x * cspace + 1 - i, 1
                else:
                    percolations.unpersist()

                    self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")

                rdd = self._sc.range(
                    self._edges
                ).flatMap(
                    __map
                )
            else:
                percolations.unpersist()

                self._logger.error("invalid percolations generation mode")
                raise ValueError("invalid percolations generation mode")
        else:
            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(x):
                    for i in range(cspace):
                        l = (-1) ** i
                        yield i * pspace + (x + l) % pspace, i * pspace + x, 1
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(x):
                    for i in range(cspace):
                        l = (-1) ** i
                        yield ((x + l) % pspace) * cspace + i, x * cspace + i, 1
            else:
                percolations.unpersist()

                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            rdd = self._sc.range(
                pspace
            ).flatMap(
                __map
            )

        return Operator(rdd, shape, dtype=int, nelem=nelem)
