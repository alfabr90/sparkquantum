import unittest

from pyspark import SparkContext


class Base(unittest.TestCase):
    def setUp(self):
        self.spark_context = SparkContext.getOrCreate()
        self.spark_context.setLogLevel('ERROR')
