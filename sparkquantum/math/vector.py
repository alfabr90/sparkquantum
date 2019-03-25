import math
import numpy as np

from pyspark.sql import functions

from sparkquantum.math.base import Base
from sparkquantum.utils.utils import Utils

__all__ = ['Vector', 'is_vector']


class Vector(Base):
    """Class for vectors."""

    def __init__(self, df, shape, data_type=complex):
        """Build a `Vector` object.

        Parameters
        ----------
        df : `DataFrame`
            The base DataFrame of this object.
        shape : tuple
            The shape of this vector object. Must be a 2-dimensional tuple.
        data_type : type, optional
            The Python type of all values in this object. Default value is complex.

        """
        super().__init__(rdd, shape, data_type=data_type)

    def dump(self, path, glue=None, codec=None):
        """Dump this object's DataFrame to disk in many part-* files.

        Parameters
        ----------
        path : str
            The path where the dumped DataFrame will be located at.
        glue : str, optional
            The glue string that connects each coordinate and value of each element in the DataFrame.
            Default value is `None`. In this case, it uses the 'quantum.dumpingGlue' configuration value.
        codec : str, optional
            Codec name used to compress the dumped data.
            Default value is `None`. In this case, it uses the 'quantum.dumpingCompressionCodec' configuration value.

        """
        if glue is None:
            glue = Utils.get_conf(self._spark_session, 'quantum.dumpingGlue')

        if codec is None:
            codec = Utils.get_conf(self._spark_session, 'quantum.dumpingCompressionCodec')

        df = self.data.map(
            lambda m: glue.join((str(m[0]), str(m[1])))
        )

        df.saveAsTextFile(path, codec)

    def numpy_array(self):
        """Create a numpy array containing this object's DataFrame data.

        Returns
        -------
        ndarray
            The numpy array

        """
        data = self.data.collect()
        result = np.zeros(self._shape, dtype=self._data_type)

        for d in data:
            result[d['i']] = d['v']

        return result

    def kron(self, other):
        """Perform a tensor (Kronecker) product with another vector.

        Parameters
        ----------
        other : `Vector`
            The other vector.

        Returns
        -------
        `Vector`
            The resulting vector.

        """
        if not is_vector(other):
            if self._logger:
                self._logger.error("'Vector' instance expected, not '{}'".format(type(other)))
            raise TypeError("'Vector' instance expected, not '{}'".format(type(other)))

        other_shape = other.shape
        new_shape = (self._shape[0] * other_shape[0], 1)
        data_type = Utils.get_precendent_type(self._data_type, other.data_type)

        a = self.data.select(
            functions.col('0').alias('a'), self.data['i'].alias('a_i'), self.data['v'].alias('a_v')
        )

        b = other.data.select(
            functions.col('0').alias('b'), self.data['i'].alias('b_i'), self.data['v'].alias('b_v')
        )

        df = a.join(
            b, a['a'] == b['b'], 'inner'
        ).select(
            a['a_i'] * other_shape[0] + b['b_i'], a['a_v'] * b['b_v']
        )

        return Vector(df, new_shape, data_type=data_type)

    def norm(self):
        """Calculate the norm of this vector.

        Returns
        -------
        float
            The norm of this vector.

        """
        n = self.data.filter(
            self.data['v'] != self._data_type()
        ).agg(
            functions.sum(functions.abs(self.data['v']) ** 2).alias('v')
        ).take(1)[0]['v']

        if n is None:
            return 0
        else:
            return math.sqrt(n)

    def is_unitary(self):
        """Check if this vector is unitary by calculating its norm.

        Returns
        -------
        bool
            True if the norm of this vector is 1.0, False otherwise.

        """
        round_precision = int(Utils.get_conf(self._spark_session, 'quantum.math.roundPrecision'))

        return round(self.norm(), round_precision) == 1.0


def is_vector(obj):
    """Check whether argument is a `Vector` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a `Vector` object, False otherwise.

    """
    return isinstance(obj, Vector)
