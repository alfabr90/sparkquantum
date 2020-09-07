import math

from pyspark import SparkContext, SparkConf

from sparkquantum import constants, plot, util
from sparkquantum.dtqw.coin.coin1d.hadamard import Hadamard
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.dtqw.gauge.position import PositionGauge
from sparkquantum.dtqw.mesh.broken_links.random import RandomBrokenLinks
from sparkquantum.dtqw.mesh.mesh1d.line import Line
from sparkquantum.dtqw.state import State

'''
    DTQW 1D - 1 particle
'''
base_path = './output'
num_cores = 4

num_particles = 1
steps = 30
size = 30

bl_prob = 0.3

# Choosing a directory to store plots and logs
walk_path = "{}/{}_{}_{}_{}_{}/".format(
    base_path, 'Line', 2 * size + 1, bl_prob, steps, num_particles
)

util.create_dir(walk_path)

representation_format = constants.StateRepresentationFormatCoinPosition
# representation_format = constants.StateRepresentationFormatPositionCoin

# Initiallizing the SparkContext with some options
sparkConf = SparkConf().set(
    'sparkquantum.cluster.totalCores', num_cores
).set(
    'sparkquantum.dtqw.state.representationFormat', representation_format
)
sparkContext = SparkContext(conf=sparkConf)
sparkContext.setLogLevel('ERROR')

# Choosing a broken links generator
broken_links = RandomBrokenLinks(bl_prob)

# Choosing a coin and a mesh for the walk
coin = Hadamard()
mesh = Line([size], broken_links=broken_links)

mesh_size = mesh.size[0]

# Center of the mesh
positions = [mesh.center()]

# Options of initial states
# |i>|x> --> (|0>|x> - i|1>|x>) / sqrt(2)
amplitudes = [[(1.0 + 0.0j) / math.sqrt(2),
               (0.0 - 1.0j) / math.sqrt(2)]]

# |i>|x> --> |0>|x>
# amplitudes = [[(1.0 + 0.0j), 0]]

# |i>|x> --> |1>|x>
# amplitudes = [[0, (1.0 + 0.0j)]]

# Building the initial state
initial_state = State.create(
    coin,
    mesh,
    positions,
    amplitudes,
    representationFormat=representation_format)

# Instantiating the walk
dtqw = DiscreteTimeQuantumWalk(initial_state)

# Performing the walk
final_state = dtqw.walk(steps)

# Measuring the state of the system and plotting its probability distribution
gauge = PositionGauge()

joint = gauge.measure(final_state)

axis = mesh.axis()
data = joint.ndarray()
labels = [v.name for v in joint.variables] + ['Probability']

plot.line(axis, data, walk_path + 'joint_1d1p', labels=labels, dpi=300)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
sparkContext.stop()
