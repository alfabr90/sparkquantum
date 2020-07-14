import logging

from pyspark import RDD

from sparkquantum import constants

__all__ = ['get_conf']


_config_defaults = {
    'sparkquantum.cluster.maxPartitionSize': 64 * 10 ** 6,
    'sparkquantum.cluster.numPartitionsSafetyFactor': 1.3,
    'sparkquantum.cluster.totalCores': 1,
    'sparkquantum.cluster.useSparkDefaultNumPartitions': 'False',
    'sparkquantum.dtqw.interactionOperator.checkpoint': 'False',
    'sparkquantum.dtqw.mesh.brokenLinks.generationMode': constants.BrokenLinksGenerationModeBroadcast,
    'sparkquantum.dtqw.profiler.logExecutors': 'False',
    'sparkquantum.dtqw.walk.checkpointingFrequency': -1,
    'sparkquantum.dtqw.walk.checkUnitary': 'False',
    'sparkquantum.dtqw.walk.dumpingFrequency': -1,
    'sparkquantum.dtqw.walk.dumpingPath': './',
    'sparkquantum.dtqw.walk.dumpStatesProbabilityDistributions': 'False',
    'sparkquantum.dtqw.walkOperator.checkpoint': 'False',
    'sparkquantum.dtqw.walkOperator.kroneckerMode': constants.KroneckerModeBroadcast,
    'sparkquantum.dtqw.walkOperator.tempPath': './',
    'sparkquantum.dtqw.state.dumpingFormat': constants.StateDumpingFormatIndex,
    'sparkquantum.dtqw.state.representationFormat': constants.StateRepresentationFormatCoinPosition,
    'sparkquantum.dumpingCompressionCodec': None,
    'sparkquantum.dumpingGlue': ' ',
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


def get_conf(sc, config_str):
    """Get a configuration value from the :py:class:`pyspark.SparkContext` object.

    Parameters
    ----------
    sc : :py:class:`pyspark.SparkContext`
        The :py:class:`pyspark.SparkContext` object.
    config_str : str
        The configuration string to have its correspondent value obtained.

    Returns
    -------
    str
        The configuration value or None if the configuration is not found.

    """
    c = sc.getConf().get(config_str)

    if not c:
        if config_str not in _config_defaults:
            return None
        return _config_defaults[config_str]

    return c
