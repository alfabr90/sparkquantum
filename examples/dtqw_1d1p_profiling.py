import math
import logging

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin1d.hadamard1d import Hadamard1D
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.dtqw.gauge.position_gauge import PositionGauge
from sparkquantum.dtqw.mesh.mesh1d.line import Line
from sparkquantum.dtqw.qw_profiler import QuantumWalkProfiler
from sparkquantum.dtqw.state import State
from sparkquantum.utils.utils import Utils

'''
    DTQW 1D - 1 particle
'''
base_path = './output/'
num_cores = 4
profile = True

num_particles = 1
steps = 30
size = 30

# Choosing a directory to store plots and logs
walk_path = "{}/{}_{}_{}_{}/".format(
    base_path, 'Line', 2 * size + 1, steps, num_particles
)

Utils.create_dir(walk_path)

representationFormat = Utils.StateRepresentationFormatCoinPosition
# representationFormat = Utils.StateRepresentationFormatPositionCoin

# Initiallizing the SparkContext with some options
sparkConf = SparkConf().set(
    'quantum.cluster.totalCores', num_cores
).set(
    'quantum.dtqw.state.representationFormat', representationFormat
).set(
    'quantum.logging.enabled', 'True'
).set(
    'quantum.logging.level', logging.DEBUG
).set(
    'quantum.logging.filename', walk_path + 'log.txt'
).set(
    'quantum.profiling.enabled', 'True'
)
sparkContext = SparkContext(conf=sparkConf)
sparkContext.setLogLevel('ERROR')

# Choosing a coin and a mesh for the walk
coin = Hadamard1D()
mesh = Line(size)

# Adding the profiler to the classes and starting it
profiler = QuantumWalkProfiler()

coin.profiler = profiler
mesh.profiler = profiler

mesh_size = mesh.size

# Center of the mesh
positions = (int((mesh_size - 1) / 2), )

# Options of initial states
# |i>|x> --> (|0>|0> - i|1>|0>) / sqrt(2)
amplitudes = (((1.0 + 0.0j) / math.sqrt(2), (0.0 - 1.0j) / math.sqrt(2)), )

# |i>|x> --> |0>|0>
# amplitudes = (((1.0 + 0.0j), ), )

# |i>|x> --> |1>|0>
# amplitudes = ((0, (1.0 + 0.0j)), )

# Building the initial state
initial_state = State.create(
    coin,
    mesh,
    positions,
    amplitudes,
    representationFormat)

# Instatiating the walk
dtqw = DiscreteTimeQuantumWalk(coin, mesh, num_particles)

dtqw.profiler = profiler

# Performing the walk
final_state = dtqw.walk(steps, initial_state)

final_state.profiler = profiler

# Measuring the state of the system and plotting its PDF
gauge = PositionGauge()

gauge.profiler = profiler

joint = gauge.measure(final_state)
joint.plot(walk_path + 'joint_1d1p', dpi=300)

# Exporting the profiling data
profiler.export(walk_path)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
sparkContext.stop()
