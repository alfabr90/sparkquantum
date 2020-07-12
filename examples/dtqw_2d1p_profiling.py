import math
import logging

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin2d.hadamard import Hadamard
from sparkquantum.dtqw.gauge.position_gauge import PositionGauge
from sparkquantum.dtqw.mesh.mesh2d.diagonal.lattice import Lattice
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.qw_profiler import QuantumWalkProfiler
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.utils.utils import Utils

'''
    DTQW 2D - 1 particle
'''
base_path = './output/'
num_cores = 4
profile = True

num_particles = 1
steps = 30
size = 30

# Choosing a directory to store plots and logs
walk_path = "{}/{}_{}_{}_{}/".format(
    base_path, 'DiagonalLattice', 2 * size + 1, steps, num_particles
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
coin = Hadamard()
mesh = Lattice((size, size))

mesh_size = mesh.size[0] * mesh.size[1]

# Center of the mesh
positions = [mesh.center()]

# Options of initial states
# |i,j>|x,y> --> (|0,0>|x,y> + i|0,1>|x,y> - i|1,0>|x,y> + |1,1>|x,y>) / 2
amplitudes = [[(1.0 + 0.0j) / 2,
               (0.0 + 1.0j) / 2,
               (0.0 - 1.0j) / 2,
               (1.0 + 0.0j) / 2]]

# |i,j>|x,y> --> (|0,0>|x,y> + i|0,1>|x,y> + i|1,0>|x,y> - |1,1>|x,y>) / 2
# amplitudes = [[(1.0 + 0.0j) / 2,
#                (0.0 + 1.0j) / 2,
#                (0.0 + 1.0j) / 2,
#                (-1.0 - 0.0j) / 2]]

# |i,j>|x,y> --> (|0,0>|x,y> - |0,1>|x,y> - |1,0>|x,y> + |1,1>|x,y>) / 2
# amplitudes = [[(1.0 + 0.0j) / 2,
#                (-1.0 - 0.0j) / 2,
#                (-1.0 - 0.0j) / 2,
#                (1.0 + 0.0j) / 2]]

# Building the initial state
initial_state = State.create(
    coin,
    mesh,
    positions,
    amplitudes,
    representationFormat=representationFormat)

# Instantiating the walk
dtqw = DiscreteTimeQuantumWalk(initial_state)

# Performing the walk
final_state = dtqw.walk(steps)

# Measuring the state of the system and plotting its probability distribution
gauge = PositionGauge()

joint = gauge.measure(final_state)
joint.plot(walk_path + 'joint_2d1p', dpi=300)
joint.plot_contour(walk_path + 'joint_2d1p_contour', dpi=300)

# Exporting the profiling data
dtqw.profiler.export(walk_path)
gauge.profiler.export(walk_path)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
sparkContext.stop()
