import logging

from sparkquantum import constants

__all__ = ['get']


_defaults = {
    'sparkquantum.cluster.maxPartitionSize': 64 * 10 ** 6,
    'sparkquantum.cluster.numPartitionsSafetyFactor': 1.3,
    'sparkquantum.cluster.totalCores': 1,
    'sparkquantum.cluster.useSparkDefaultNumPartitions': 'False',
    'sparkquantum.dtqw.checkpointingFrequency': -1,
    'sparkquantum.dtqw.checkUnitary': 'False',
    'sparkquantum.dtqw.dumpingFrequency': -1,
    'sparkquantum.dtqw.dumpStatesProbabilityDistributions': 'False',
    'sparkquantum.dtqw.evolutionOperator.checkpoint': 'False',
    'sparkquantum.dtqw.evolutionOperator.kroneckerMode': constants.KroneckerModeBroadcast,
    'sparkquantum.dtqw.evolutionOperator.tempPath': './',
    'sparkquantum.dtqw.interactionOperator.checkpoint': 'False',
    'sparkquantum.dtqw.mesh.percolation.generationMode': constants.PercolationsGenerationModeBroadcast,
    'sparkquantum.dtqw.profiler.logExecutors': 'False',
    'sparkquantum.dtqw.stateRepresentationFormat': constants.StateRepresentationFormatCoinPosition,
    'sparkquantum.dtqw.state.dumpingFormat': constants.StateDumpingFormatIndex,
    'sparkquantum.dumpingCompressionCodec': None,
    'sparkquantum.dumpingGlue': ' ',
    'sparkquantum.dumpingPath': './',
    'sparkquantum.logging.enabled': 'False',
    'sparkquantum.logging.filename': './log.txt',
    'sparkquantum.logging.format': '%(levelname)s:%(name)s:%(asctime)s:%(message)s',
    'sparkquantum.logging.level': logging.WARNING,
    'sparkquantum.math.dumpingMode': constants.DumpingModePartFiles,
    'sparkquantum.math.roundPrecision': 10,
    'sparkquantum.profiling.enabled': 'False',
    'sparkquantum.profiling.baseUrl': 'http://localhost:4040/api/v1/'
}
"""
Dict with the default values for all accepted configurations of the package.
"""


def get(sc, config):
    """Get a configuration value from the :py:class:`pyspark.SparkContext` object.

    Parameters
    ----------
    sc : :py:class:`pyspark.SparkContext`
        The :py:class:`pyspark.SparkContext` object.
    config : str
        The configuration string to have its correspondent value obtained.

    Returns
    -------
    str
        The configuration value or None if the configuration is not found.

    """
    c = sc.getConf().get(config)

    if not c:
        if config not in _defaults:
            return None
        return _defaults[config]

    return c
