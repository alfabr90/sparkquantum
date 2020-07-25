import math

from pyspark import SparkContext, SparkConf

from sparkquantum import constants, plot, util
from sparkquantum.dtqw.coin.coin2d.hadamard import Hadamard
from sparkquantum.dtqw.gauge.position import PositionGauge
from sparkquantum.dtqw.mesh.mesh2d.diagonal.lattice import Lattice
from sparkquantum.dtqw.mesh.broken_links.random import RandomBrokenLinks
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.profiler import QuantumWalkProfiler
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk

'''
    DTQW 2D - 1 particle
'''
base_path = './output/'
num_cores = 4

num_particles = 1
steps = 30
size = 30

bl_prob = 0.3

# Choosing a directory to store plots and logs
walk_path = "{}/{}_{}_{}_{}_{}/".format(
    base_path, 'DiagonalLattice', 2 * size + 1, bl_prob, steps, num_particles
)

util.create_dir(walk_path)

representationFormat = constants.StateRepresentationFormatCoinPosition
# representationFormat = constants.StateRepresentationFormatPositionCoin

# Initiallizing the SparkContext with some options
sparkConf = SparkConf().set(
    'sparkquantum.cluster.totalCores', num_cores
).set(
    'sparkquantum.dtqw.state.representationFormat', representationFormat
)
sparkContext = SparkContext(conf=sparkConf)
sparkContext.setLogLevel('ERROR')

# Choosing a broken links generator
broken_links = RandomBrokenLinks(bl_prob)

# Choosing a coin and a mesh for the walk
coin = Hadamard()
mesh = Lattice((size, size), broken_links=broken_links)

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

axis = mesh.axis()
data = joint.ndarray()
labels = [v.name for v in joint.variables]

plot.surface(axis, data, walk_path + 'joint_2d1p',
             labels=labels + ['Probability'], dpi=300)
plot.contour(axis, data, walk_path + 'joint_2d1p_contour',
             labels=labels, dpi=300)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
sparkContext.stop()
