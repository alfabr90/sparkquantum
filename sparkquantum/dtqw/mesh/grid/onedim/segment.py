from sparkquantum import constants, util
from sparkquantum.dtqw.mesh.grid.onedim.onedim import OneDimensional
from sparkquantum.dtqw.operator import Operator

__all__ = ['Segment']


class Segment(OneDimensional):
    """Class for Segment grids."""

    def __init__(self, shape, percolation=None):
        """Build a Segment object.

        Parameters
        ----------
        shape : tuple of int
            Shape of this grid. Must be 1-dimensional.
        percolation : :py:class:`sparkquantum.dtqw.mesh.percolation.Percolation`, optional
            A percolation object.

        """
        super().__init__(shape, percolation=percolation)

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

        return 'Segment grid with shape {}{}'.format(self._shape, percolation)

    def create_operator(self,
                        repr_format=constants.StateRepresentationFormatCoinPosition, perc_mode=constants.PercolationsGenerationModeBroadcast):
        """Build the shift operator for a quantum walk.

        Parameters
        ----------
        repr_format : int, optional
            Indicate how the quantum system is represented.
            Default value is :py:const:`sparkquantum.constants.StateRepresentationFormatCoinPosition`.
        perc_mode : int, optional
            Indicate how the percolation will be represented.
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

        if self._percolation:
            percolation = self._percolation.generate(self._edges)

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
                                bl = 0
                            else:
                                if x + l >= pspace or x + l < 0:
                                    bl = 0
                                else:
                                    bl = l

                            yield (i + bl) * pspace + x + bl, (1 - i) * pspace + x, 1
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(cspace):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the
                            # vertex from the edge number
                            x = (e[1][0] - i - l) % pspace

                            if e[1][1]:
                                bl = 0
                            else:
                                if x + l >= pspace or x + l < 0:
                                    bl = 0
                                else:
                                    bl = l

                            yield (x + bl) * cspace + i + bl, x * cspace + 1 - i, 1
                else:
                    self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")

                rdd = self._sc.range(
                    self._edges
                ).map(
                    lambda m: (m, m)
                ).leftOuterJoin(
                    percolation
                ).flatMap(
                    __map
                )
            elif perc_mode == constants.PercolationsGenerationModeBroadcast:
                percolation = util.broadcast(
                    self._sc, percolation)

                if repr_format == constants.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        for i in range(cspace):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the
                            # vertex from the edge number
                            x = (e - i - l) % pspace

                            if e in percolation.value:
                                bl = 0
                            else:
                                if x + l >= pspace or x + l < 0:
                                    bl = 0
                                else:
                                    bl = l

                            yield (i + bl) * pspace + x + bl, (1 - i) * pspace + x, 1
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        for i in range(cspace):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the
                            # vertex from the edge number
                            x = (e - i - l) % pspace

                            if e in percolation.value:
                                bl = 0
                            else:
                                if x + l >= pspace or x + l < 0:
                                    bl = 0
                                else:
                                    bl = l

                            yield (x + bl) * cspace + i + bl, x * cspace + 1 - i, 1
                else:
                    percolation.unpersist()

                    self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")

                rdd = self._sc.range(
                    self._edges
                ).flatMap(
                    __map
                )
            else:
                self._logger.error("invalid percolation generation mode")
                raise ValueError("invalid percolation generation mode")
        else:
            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(x):
                    for i in range(cspace):
                        l = (-1) ** i

                        if x + l >= pspace or x + l < 0:
                            bl = 0
                        else:
                            bl = l

                        yield (i + bl) * pspace + x + bl, (1 - i) * pspace + x, 1
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(x):
                    for i in range(cspace):
                        l = (-1) ** i

                        if x + l >= pspace or x + l < 0:
                            bl = 0
                        else:
                            bl = l

                        yield (x + bl) * cspace + i + bl, x * cspace + (1 - i), 1
            else:
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            rdd = self._sc.range(
                pspace
            ).flatMap(
                __map
            )

        return Operator(rdd, shape, dtype=int, nelem=nelem)
