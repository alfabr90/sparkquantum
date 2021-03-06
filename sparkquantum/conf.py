import logging

from sparkquantum import constants

__all__ = ['get']


_defaults = {
    'sparkquantum.cluster.totalCores': 1,
    'sparkquantum.logging.enabled': False,
    'sparkquantum.logging.filename': './log.txt',
    'sparkquantum.logging.format': '%(levelname)s:%(name)s:%(asctime)s:%(message)s',
    'sparkquantum.logging.level': logging.WARNING,
    'sparkquantum.math.roundPrecision': 10,
    'sparkquantum.partitioning.enabled': True,
    'sparkquantum.partitioning.rddPartitionSize': 32 * 1024 ** 2,
    'sparkquantum.partitioning.safetyFactor': 1.3,
    'sparkquantum.profiling.enabled': False
}
"""
Dict with the default values for all accepted configurations of the package.
"""


def get(sc, config, default=None):
    """Get a configuration value from the :py:class:`pyspark.SparkContext` object.

    Parameters
    ----------
    sc : :py:class:`pyspark.SparkContext`
        The :py:class:`pyspark.SparkContext` object.
    config : str
        The configuration string to have its correspondent value obtained.
    default: any
        The default value for the configuration.

    Returns
    -------
    any
        The configuration value.

    """
    c = sc.getConf().get(config)

    if c is None:
        if default is not None:
            c = default
        elif config in _defaults:
            c = _defaults[config]

    return c
