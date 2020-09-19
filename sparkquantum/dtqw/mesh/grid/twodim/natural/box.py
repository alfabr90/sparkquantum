from sparkquantum import constants, util
from sparkquantum.dtqw.mesh.grid.twodim.natural.natural import Natural
from sparkquantum.dtqw.operator import Operator

__all__ = ['Box']


class Box(Natural):
    """Class for natural box mesh."""

    def __init__(self, shape, percolation=None):
        """Build a natural box object.

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

        return 'Natural box with shape {}{}'.format(
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
        csubspace = 2
        cspace = csubspace ** self._ndim
        psubspace = self._shape
        pspace = self._sites
        shape = (cspace * pspace, cspace * pspace)

        nelem = shape[0]

        if self._percolation is not None:
            percolation = self._percolation.generate(self._edges)

            if perc_mode == constants.PercolationsGenerationModeRDD:
                if repr_format == constants.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(csubspace):
                            l = (-1) ** i

                            # Finding the correspondent x,y coordinates of the
                            # vertex from the edge number
                            if e[1][0] >= pspace:
                                j = i
                                x = int(
                                    (e[1][0] - pspace) / psubspace[0])
                                y = ((e[1][0] - pspace) %
                                     psubspace[1] - i - l) % psubspace[1]
                            else:
                                j = int(not i)
                                x = (
                                    e[1][0] % psubspace[0] - i - l) % psubspace[0]
                                y = int(e[1][0] / psubspace[0])

                            delta = int(not (i ^ j))
                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            if e[1][1]:
                                l = 0
                            else:
                                # The border edges are considered broken so
                                # that they become reflexive
                                if pos1 >= psubspace[0] or pos1 < 0 or pos2 >= psubspace[1] or pos2 < 0:
                                    l = 0

                            m = ((i + l) * csubspace + (abs(j + l) % csubspace)) * pspace + \
                                (x + l * (1 - delta)) * \
                                psubspace[1] + (y + l * delta)
                            n = ((1 - i) * csubspace + (1 - j)) * \
                                pspace + x * psubspace[1] + y

                            yield m, n, 1
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(csubspace):
                            l = (-1) ** i

                            # Finding the correspondent x,y coordinates of the
                            # vertex from the edge number
                            if e[1][0] >= pspace:
                                j = i
                                x = int(
                                    (e[1][0] - pspace) / psubspace[0])
                                y = ((e[1][0] - pspace) %
                                     psubspace[1] - i - l) % psubspace[1]
                            else:
                                j = int(not i)
                                x = (
                                    e[1][0] % psubspace[0] - i - l) % psubspace[0]
                                y = int(e[1][0] / psubspace[0])

                            delta = int(not (i ^ j))
                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            if e[1][1]:
                                l = 0
                            else:
                                # The border edges are considered broken so
                                # that they become reflexive
                                if pos1 >= psubspace[0] or pos1 < 0 or pos2 >= psubspace[1] or pos2 < 0:
                                    l = 0

                            m = ((x + l * (1 - delta)) * psubspace[1] + (y + l * delta)) * cspace + \
                                (i + l) * csubspace + \
                                (abs(j + l) % csubspace)
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
            elif perc_mode == constants.PercolationsGenerationModeBroadcast:
                percolation = util.broadcast(percolation)

                if repr_format == constants.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(csubspace):
                            l = (-1) ** i

                            # Finding the correspondent x,y coordinates of the
                            # vertex from the edge number
                            if e >= pspace:
                                j = i
                                delta = int(not (i ^ j))
                                x = int((e - pspace) / psubspace[0])
                                y = ((e - pspace) %
                                     psubspace[1] - i - l) % psubspace[1]
                            else:
                                j = int(not i)
                                delta = int(not (i ^ j))
                                x = (e % psubspace[0] - i - l) % psubspace[0]
                                y = int(e / psubspace[0])

                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            if e in percolation.value:
                                bl = 0
                            else:
                                # The border edges are considered broken so
                                # that they become reflexive
                                if pos1 >= psubspace[0] or pos1 < 0 or pos2 >= psubspace[1] or pos2 < 0:
                                    bl = 0
                                else:
                                    bl = l

                            m = ((i + bl) * csubspace + (abs(j + bl) % csubspace)) * pspace + \
                                (x + bl * (1 - delta)) * \
                                psubspace[1] + (y + bl * delta)
                            n = ((1 - i) * csubspace + (1 - j)) * \
                                pspace + x * psubspace[1] + y

                            yield m, n, 1
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(csubspace):
                            l = (-1) ** i

                            # Finding the correspondent x,y coordinates of the
                            # vertex from the edge number
                            if e >= pspace:
                                j = i
                                delta = int(not (i ^ j))
                                x = int(
                                    (e - pspace) / psubspace[0])
                                y = ((e - pspace) %
                                     psubspace[1] - i - l) % psubspace[1]
                            else:
                                j = int(not i)
                                delta = int(not (i ^ j))
                                x = (e % psubspace[0] - i - l) % psubspace[0]
                                y = int(e / psubspace[0])

                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            if e in percolation.value:
                                bl = 0
                            else:
                                # The border edges are considered broken so
                                # that they become reflexive
                                if pos1 >= psubspace[0] or pos1 < 0 or pos2 >= psubspace[1] or pos2 < 0:
                                    bl = 0
                                else:
                                    bl = l

                            m = ((x + bl * (1 - delta)) * psubspace[1] + (y + bl * delta)) * cspace + \
                                (i + bl) * csubspace + \
                                (abs(j + bl) % csubspace)
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
                        l = (-1) ** i
                        for j in range(csubspace):
                            delta = int(not (i ^ j))

                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            # The border edges are considered broken so that
                            # they become reflexive
                            if pos1 >= psubspace[0] or pos1 < 0 or pos2 >= psubspace[1] or pos2 < 0:
                                bl = 0
                            else:
                                bl = l

                            m = ((i + bl) * csubspace + (abs(j + bl) % csubspace)) * pspace + \
                                (x + bl * (1 - delta)) * \
                                psubspace[1] + (y + bl * delta)
                            n = ((1 - i) * csubspace + (1 - j)) * \
                                pspace + x * psubspace[1] + y

                            yield m, n, 1
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(xy):
                    x = xy % psubspace[0]
                    y = int(xy / psubspace[0])

                    for i in range(csubspace):
                        l = (-1) ** i
                        for j in range(csubspace):
                            delta = int(not (i ^ j))

                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            # The border edges are considered broken so that
                            # they become reflexive
                            if pos1 >= psubspace[0] or pos1 < 0 or pos2 >= psubspace[1] or pos2 < 0:
                                bl = 0
                            else:
                                bl = l

                            m = ((x + bl * (1 - delta)) * psubspace[1] + (y + bl * delta)) * cspace + \
                                (i + bl) * csubspace + \
                                (abs(j + bl) % csubspace)
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
