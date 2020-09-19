import unittest

from pyspark import SparkContext


class Base(unittest.TestCase):
    def setUp(self):
        self.sc = SparkContext.getOrCreate()
        self.sc.setLogLevel('ERROR')
