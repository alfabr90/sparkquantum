import math

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin2d.hadamard import Hadamard
from sparkquantum.dtqw.gauge.position_gauge import PositionGauge
from sparkquantum.dtqw.mesh.mesh2d.diagonal.lattice import Lattice
from sparkquantum.dtqw.mesh.broken_links.random_broken_links import RandomBrokenLinks
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.qw_profiler import QuantumWalkProfiler
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.utils.utils import Utils

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
mesh = Lattice((size, size), broken_links=broken_links)

mesh_size = mesh.size[0] * mesh.size[1]

# Center of the mesh
positions = (int((mesh.size[0] - 1) / 2) *
             mesh.size[1] + int((mesh.size[1] - 1) / 2), )

# Options of initial states
# |i,j>|x,y> --> (|0,0>|0,0> + i|0,1>|0,0> - i|1,0>|0,0> + |1,1>|0,0>) / 2
amplitudes = ((((1.0 + 0.0j) / 2),
               ((0.0 + 1.0j) / 2),
               ((0.0 - 1.0j) / 2),
               ((1.0 + 0.0j) / 2)), )

# |i,j>|x,y> --> (|0,0>|0,0> + i|0,1>|0,0> + i|1,0>|0,0> - |1,1>|0,0>) / 2
# amplitudes = ((((1.0 + 0.0j) / 2),
#               ((0.0 + 1.0j) / 2),
#               ((0.0 + 1.0j) / 2),
#               ((-1.0 - 0.0j) / 2)), )

# |i,j>|x,y> --> (|0,0>|0,0> - |0,1>|0,0> - |1,0>|0,0> + |1,1>|0,0>) / 2
# amplitudes = ((((1.0 + 0.0j) / 2),
#               ((-1.0 - 0.0j) / 2),
#               ((-1.0 - 0.0j) / 2),
#               ((1.0 + 0.0j) / 2)), )

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
joint.plot(walk_path + 'joint_2d1p', dpi=300)
joint.plot_contour(walk_path + 'joint_2d1p_contour', dpi=300)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
sparkContext.stop()
