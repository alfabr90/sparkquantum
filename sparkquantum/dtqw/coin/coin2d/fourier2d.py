import numpy as np

from sparkquantum.dtqw.coin.coin2d.coin2d import Coin2D

__all__ = ['Fourier2D']


class Fourier2D(Coin2D):
    """Class that represents the 2-dimensional Fourier coin."""

    def __init__(self, spark_session):
        """Build a 2-dimensional Fourier `Coin` object.

        Parameters
        ----------
        spark_session : `SparkSession`
            The `SparkSession` object.

        """
        super().__init__(spark_session)

        self._data = np.array(
            [[1, 1, 1, 1],
             [1, 1.0j, -1, -1.0j],
             [1, -1, 1, -1],
             [1, -1.0j, -1, 1.0j]], dtype=complex
        ) / 2.0
