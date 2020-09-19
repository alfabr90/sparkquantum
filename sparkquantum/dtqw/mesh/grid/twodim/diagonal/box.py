from sparkquantum import constants, util
from sparkquantum.dtqw.mesh.grid.twodim.diagonal.diagonal import Diagonal
from sparkquantum.dtqw.operator import Operator

__all__ = ['Box']


class Box(Diagonal):
    """Class for diagonal box mesh."""

    def __init__(self, shape, percolation=None):
        """Build a diagonal box object.

        Parameters
        ----------
        shape : tuple of int
            Shape of this grid. Must be 2-dimensional.
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

        return 'Diagonal box with shape {}{}'.format(
            self._shape, percolation)

    def create_operator(self,
                        repr_format=constants.StateRepresentationFormatCoinPosition, perc_mode=constants.PercolationGenerationModeBroadcast):
        """Build the shift operator for a quantum walk.

        Parameters
        ----------
        repr_format : int, optional
            Indicate how the quantum system is represented.
            Default value is :py:const:`sparkquantum.constants.StateRepresentationFormatCoinPosition`.
        perc_mode : int, optional
            Indicate how the percolation will be represented.
            Default value is :py:const:`sparkquantum.constants.PercolationGenerationModeBroadcast`.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator`
            The created operator using this mesh.

        Raises
        ------
        ValueError
            If `repr_format` or `perc_mode` is not valid.

        """
        csubspace = 2
        cspace = csubspace ** self._ndim
        psubspace = self._shape
        pspace = self._sites
        shape = (cspace * pspace, cspace * pspace)

        nelem = shape[0]

        if self._percolation is not None:
            percolation = self._percolation.generate(self._edges)

            if perc_mode == constants.PercolationGenerationModeRDD:
                if repr_format == constants.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(csubspace):
                            l1 = (-1) ** i
                            for j in range(csubspace):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of
                                # the vertex from the edge number
                                x = (
                                    e[1][0] % psubspace[0] - i - l1) % psubspace[0]
                                y = (int(e[1][0] / psubspace[0]) -
                                     j - l2) % psubspace[1]

                                if e[1][1]:
                                    bl1, bl2 = 0, 0
                                else:
                                    # The border edges are considered broken so
                                    # that they become reflexive
                                    if x + \
                                            l1 >= psubspace[0] or x + l1 < 0 or y + l2 >= psubspace[1] or y + l2 < 0:
                                        bl1, bl2 = 0, 0
                                    else:
                                        bl1, bl2 = l1, l2

                                m = ((i + bl1) * csubspace + (j + bl2)) * \
                                    pspace + (x + bl1) * \
                                    psubspace[1] + (y + bl2)
                                n = ((1 - i) * csubspace + (1 - j)) * \
                                    pspace + x * psubspace[1] + y

                                yield m, n, 1
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(csubspace):
                            l1 = (-1) ** i
                            for j in range(csubspace):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of
                                # the vertex from the edge number
                                x = (
                                    e[1][0] % psubspace[0] - i - l1) % psubspace[0]
                                y = (int(e[1][0] / psubspace[0]) -
                                     j - l2) % psubspace[1]

                                if e[1][1]:
                                    bl1, bl2 = 0, 0
                                else:
                                    # The border edges are considered broken so
                                    # that they become reflexive
                                    if x + \
                                            l1 >= psubspace[0] or x + l1 < 0 or y + l2 >= psubspace[1] or y + l2 < 0:
                                        bl1, bl2 = 0, 0
                                    else:
                                        bl1, bl2 = l1, l2

                                m = (
                                    (x + bl1) * psubspace[1] + (y + bl2)) * cspace + (i + bl1) * csubspace + (j + bl2)
                                n = (x * psubspace[1] + y) * cspace + \
                                    (1 - i) * csubspace + (1 - j)

                                yield m, n, 1
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
            elif perc_mode == constants.PercolationGenerationModeBroadcast:
                percolation = util.broadcast(percolation)

                if repr_format == constants.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        for i in range(csubspace):
                            l1 = (-1) ** i
                            for j in range(csubspace):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of
                                # the vertex from the edge number
                                x = (e % psubspace[0] - i - l1) % psubspace[0]
                                y = (
                                    int(e / psubspace[0]) - j - l2) % psubspace[1]

                                if e in percolation.value:
                                    bl1, bl2 = 0, 0
                                else:
                                    # The border edges are considered broken so
                                    # that they become reflexive
                                    if x + \
                                            l1 >= psubspace[0] or x + l1 < 0 or y + l2 >= psubspace[1] or y + l2 < 0:
                                        bl1, bl2 = 0, 0
                                    else:
                                        bl1, bl2 = l1, l2

                                m = ((i + bl1) * csubspace + (j + bl2)) * \
                                    pspace + (x + bl1) * \
                                    psubspace[1] + (y + bl2)
                                n = ((1 - i) * csubspace + (1 - j)) * \
                                    pspace + x * psubspace[1] + y

                                yield m, n, 1
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        for i in range(csubspace):
                            l1 = (-1) ** i
                            for j in range(csubspace):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of
                                # the vertex from the edge number
                                x = (e % psubspace[0] - i - l1) % psubspace[0]
                                y = (
                                    int(e / psubspace[0]) - j - l2) % psubspace[1]

                                if e in percolation.value:
                                    bl1, bl2 = 0, 0
                                else:
                                    # The border edges are considered broken so
                                    # that they become reflexive
                                    if x + \
                                            l1 >= psubspace[0] or x + l1 < 0 or y + l2 >= psubspace[1] or y + l2 < 0:
                                        bl1, bl2 = 0, 0
                                    else:
                                        bl1, bl2 = l1, l2

                                m = (
                                    (x + bl1) * psubspace[1] + (y + bl2)) * cspace + (i + bl1) * csubspace + (j + bl2)
                                n = (x * psubspace[1] + y) * cspace + \
                                    (1 - i) * csubspace + (1 - j)

                                yield m, n, 1
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
                def __map(xy):
                    x = xy % psubspace[0]
                    y = int(xy / psubspace[0])

                    for i in range(csubspace):
                        l1 = (-1) ** i
                        for j in range(csubspace):
                            l2 = (-1) ** j

                            # The border edges are considered broken so that
                            # they become reflexive
                            if x + l1 >= psubspace[0] or x + l1 < 0 or y + \
                                    l2 >= psubspace[1] or y + l2 < 0:
                                bl1, bl2 = 0, 0
                            else:
                                bl1, bl2 = l1, l2

                            m = ((i + bl1) * csubspace + (j + bl2)) * \
                                pspace + (x + bl1) * psubspace[1] + y + bl2
                            n = ((1 - i) * csubspace + (1 - j)) * \
                                pspace + x * psubspace[1] + y

                            yield m, n, 1
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(xy):
                    x = xy % psubspace[0]
                    y = int(xy / psubspace[0])

                    for i in range(csubspace):
                        l1 = (-1) ** i
                        for j in range(csubspace):
                            l2 = (-1) ** j

                            # The border edges are considered broken so that
                            # they become reflexive
                            if x + l1 >= psubspace[0] or x + l1 < 0 or y + \
                                    l2 >= psubspace[1] or y + l2 < 0:
                                bl1, bl2 = 0, 0
                            else:
                                bl1, bl2 = l1, l2

                            m = ((x + bl1) * psubspace[1] + (y + bl2)) * \
                                cspace + (i + bl1) * \
                                csubspace + (j + bl2)
                            n = (x * psubspace[1] + y) * cspace + \
                                (1 - i) * csubspace + (1 - j)

                            yield m, n, 1
            else:
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            rdd = self._sc.range(
                pspace
            ).flatMap(
                __map
            )

        return Operator(rdd, shape, dtype=int, nelem=nelem)
