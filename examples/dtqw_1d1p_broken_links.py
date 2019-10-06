import math

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin1d.hadamard1d import Hadamard1D
from sparkquantum.dtqw.mesh.mesh1d.line import Line
from sparkquantum.dtqw.mesh.broken_links.random_broken_links import RandomBrokenLinks
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.utils.utils import Utils

'''
    DTQW 1D - 1 particle
'''
base_path = './output/'
num_cores = 4

num_particles = 1
steps = 30
size = 30

bl_prob = 0.3

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
coin = Hadamard1D()
mesh = Line(size, broken_links=broken_links)

# Choosing a directory to store plots and logs
walk_path = "{}/".format(
    base_path + Utils.filename(
        mesh.filename(), steps, num_particles
    )
)

sim_path = walk_path
Utils.create_dir(sim_path)

coin_size = coin.size
mesh_size = mesh.size

# Center of the mesh
position = int((mesh_size - 1) / 2)

# Options of initial states
if representationFormat == Utils.StateRepresentationFormatCoinPosition:
    # |i>|x> --> (|0>|0> - i|1>|0>) / sqrt(2)
    state = (
        (0 * mesh_size + position, (1.0 + 0.0j) / math.sqrt(2)),
        (1 * mesh_size + position, (0.0 - 1.0j) / math.sqrt(2))
    )
    # |i>|x> --> |0>|0>
    # state = ((0 * mesh_size + position, (1.0 + 0.0j)), )
    # |i>|x> --> |1>|0>
    # state = ((1 * mesh_size + position, (1.0 + 0.0j)), )
elif representationFormat == Utils.StateRepresentationFormatPositionCoin:
    # |x>|i> --> (|0>|0> - i|0>|1>) / sqrt(2)
    state = (
        (position * coin_size + 0, (1.0 + 0.0j) / math.sqrt(2)),
        (position * coin_size + 1, (0.0 - 1.0j) / math.sqrt(2))
    )
    # |x>|i> --> |0>|0>
    # state = ((position * coin_size + 0, (1.0 + 0.0j)), )
    # |x>|i> --> |0>|1>
    # state = ((position * coin_size + 1, (1.0 + 0.0j)), )

# Building the initial state
rdd = sparkContext.parallelize(state)
shape = (coin_size * mesh_size, 1)
initial_state = State(rdd, shape, mesh, num_particles)

# Instatiating the walk
dtqw = DiscreteTimeQuantumWalk(coin, mesh, num_particles)

# Performing the walk
final_state = dtqw.walk(steps, initial_state)

# Measuring the state of the system and plotting its PDF
joint = final_state.measure()
joint.plot(sim_path + 'joint_1d1p', dpi=300)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
sparkContext.stop()
