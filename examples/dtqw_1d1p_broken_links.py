import math

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin1d.hadamard import Hadamard
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.dtqw.gauge.position_gauge import PositionGauge
from sparkquantum.dtqw.mesh.broken_links.random_broken_links import RandomBrokenLinks
from sparkquantum.dtqw.mesh.mesh1d.line import Line
from sparkquantum.dtqw.state import State
from sparkquantum.utils.utils import Utils

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

Utils.create_dir(walk_path)

representationFormat = Utils.StateRepresentationFormatCoinPosition
# representationFormat = Utils.StateRepresentationFormatPositionCoin

# Initiallizing the SparkContext with some options
sparkConf = SparkConf().set(
    'quantum.cluster.totalCores', num_cores
).set(
    'quantum.dtqw.state.representationFormat', representationFormat
)
sparkContext = SparkContext(conf=sparkConf)
sparkContext.setLogLevel('ERROR')

# Choosing a broken links generator
broken_links = RandomBrokenLinks(bl_prob)

# Choosing a coin and a mesh for the walk
coin = Hadamard()
mesh = Line(size, broken_links=broken_links)

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

# Performing the walk
final_state = dtqw.walk(steps, initial_state)

# Measuring the state of the system and plotting its PDF
gauge = PositionGauge()

joint = gauge.measure(final_state)
joint.plot(walk_path + 'joint_1d1p', dpi=300)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
sparkContext.stop()
