__all__ = [
    'DumpingModeUniqueFile',
    'DumpingModePartFiles',
    'KroneckerModeBroadcast',
    'KroneckerModeDump',
    'MatrixCoordinateDefault',
    'MatrixCoordinateMultiplier',
    'MatrixCoordinateMultiplicand',
    'MatrixCoordinateIndexed',
    'StateRepresentationFormatCoinPosition',
    'StateRepresentationFormatPositionCoin',
    'StateDumpingFormatIndex',
    'StateDumpingFormatCoordinate',
    'BrokenLinksGenerationModeBroadcast',
    'BrokenLinksGenerationModeRDD']

DumpingModeUniqueFile = 0
"""
Indicate that the mathematical entity will be dumped to a unique file in disk. This is not suitable for large working sets,
as all data must be collected to the driver node of the cluster and may not fit into memory.
"""
DumpingModePartFiles = 1
"""
Indicate that the mathematical entity will be dumped to part-* files in disk. This is done in a parallel/distributed way.
This is the default behaviour.
"""
KroneckerModeBroadcast = 0
"""
Indicate that the kronecker operation will be performed with the right operand in a broadcast variable.
Suitable for small working sets, providing the best performance due to all the data being located
locally in every node of the cluster.
"""
KroneckerModeDump = 1
"""
Indicate that the kronecker operation will be performed with the right operand previously dumped to disk.
Suitable for working sets that do not fit in cache provided by Spark.
"""
MatrixCoordinateDefault = 0
"""
Indicate that the :py:class:`sparkquantum.math.matrix.Matrix` object must have its entries stored as ``(i,j,value)`` coordinates.
"""
MatrixCoordinateMultiplier = 1
"""
Indicate that the :py:class:`sparkquantum.math.matrix.Matrix` object must have its entries stored as ``(j,(i,value))`` coordinates. This is mandatory
when the object is the multiplier operand.
"""
MatrixCoordinateMultiplicand = 2
"""
Indicate that the :py:class:`sparkquantum.math.matrix.Matrix` object must have its entries stored as ``(i,(j,value))`` coordinates. This is mandatory
when the object is the multiplicand operand.
"""
MatrixCoordinateIndexed = 3
"""
Indicate that the :py:class:`sparkquantum.math.matrix.Matrix` object must have its entries stored as ``((i,j),value)`` coordinates. This is mandatory
when the object is the multiplicand operand.
"""
StateRepresentationFormatCoinPosition = 0
"""
Indicate that the quantum system is represented as the kronecker product between the coin and position subspaces.
"""
StateRepresentationFormatPositionCoin = 1
"""
Indicate that the quantum system is represented as the kronecker product between the position and coin subspaces.
"""
StateDumpingFormatIndex = 0
"""
Indicate that the quantum system will be dumped to disk with the format ``(i,value)``.
"""
StateDumpingFormatCoordinate = 1
"""
Indicate that the quantum system will be dumped to disk with the format ``(i1,x1,...,in,xn,value)``,
for one-dimensional meshes, or ``(i1,j1,x1,y1,...,in,jn,xn,yn,value)``, for two-dimensional meshes,
for example, where n is the number of particles.
"""
BrokenLinksGenerationModeBroadcast = 0
"""
Indicate that the broken links/mesh percolation will be generated by a broadcast variable
containing the edges that are broken. Suitable for small meshes, providing the best performance
due to all the data being located locally in every node of the cluster.
"""
BrokenLinksGenerationModeRDD = 1
"""
Indicate that the broken links/mesh percolation will be generated by a RDD containing the edges that are broken.
Suitable for meshes that their corresponding data do not fit in the cache provided by Spark. Also, with this mode,
a left join is performed, making it not the best mode.
"""
