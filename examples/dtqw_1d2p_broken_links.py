import math
import cmath

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin1d.hadamard1d import Hadamard1D
from sparkquantum.dtqw.mesh.mesh1d.line import Line
from sparkquantum.dtqw.mesh.broken_links.random_broken_links import RandomBrokenLinks
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.qw_profiler import QuantumWalkProfiler
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.utils.utils import Utils
from sparkquantum.utils.logger import Logger

'''
    DTQW 1D - 2 particles
'''
base_path = './output/'
num_cores = 4

num_particles = 2
steps = 30
size = 30
entangled = True
phase = 1.0 * cmath.pi

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

# Adding a directory to store plots and logs
if entangled:
    walk_path = "{}_{}_{}/".format(
        base_path + Utils.filename(
            mesh.filename(), steps, num_particles
        ), phase, 'entangled'
    )
else:
    walk_path = "{}_{}/".format(
        base_path + Utils.filename(
            mesh.filename(), steps, num_particles
        ), phase
    )

sim_path = walk_path
Utils.create_dir(sim_path)

coin_size = coin.size
mesh_size = mesh.size

# Options of initial states
if not entangled:
    # Center of the mesh
    positions = (int((mesh_size - 1) / 2), int((mesh_size - 1) / 2))

    amplitudes = []

    # |i>|x> --> (|0>|0> - i|1>|0>) / sqrt(2)
    # amplitudes.append(
    #     ((1.0 + 0.0j) / math.sqrt(2),
    #      (0.0 - 1.0j) / math.sqrt(2)))

    # |i>|x> --> |0>|0>
    amplitudes.append(((1.0 + 0.0j), ))

    # |i>|x> --> |1>|0>
    # amplitudes.append((0, (1.0 + 0.0j)))

    # |i>|x> --> (|0>|0> - i|1>|0>) / sqrt(2)
    # amplitudes.append(
    #     ((1.0 + 0.0j) / math.sqrt(2),
    #      (0.0 - 1.0j) / math.sqrt(2)))

    # |i>|x> --> |0>|0>
    # amplitudes.append(((1.0 + 0.0j), ))

    # |i>|x> --> |1>|0>
    amplitudes.append((0, (1.0 + 0.0j)))

    initial_state = State.create(
        coin,
        mesh,
        positions,
        amplitudes,
        representationFormat)
else:
    # Center of the mesh
    position = int((mesh_size - 1) / 2)

    if representationFormat == Utils.StateRepresentationFormatCoinPosition:
        # |i1>|x1>|i2>|x2> --> (|1>|0>|0>|0> - |0>|0>|1>|0>) / sqrt(2)
        state = (
            ((1 * mesh_size + position) * coin_size * mesh_size +
             0 * mesh_size + position, 1.0 / math.sqrt(2)),
            ((0 * mesh_size + position) * coin_size * mesh_size +
             1 * mesh_size + position, -1.0 / math.sqrt(2)),
        )
    elif representationFormat == Utils.StateRepresentationFormatPositionCoin:
        # |x1>|i1>|x2>|i2> --> (|0>|1>|0>|0> - |0>|0>|0>|1>) / sqrt(2)
        state = (
            ((position * coin_size + 1) * mesh_size * coin_size +
             position * coin_size + 0, 1.0 / math.sqrt(2)),
            ((position * coin_size + 0) * mesh_size * coin_size +
             position * coin_size + 1, -1.0 / math.sqrt(2)),
        )

    rdd = sparkContext.parallelize(state)
    shape = ((coin_size * mesh_size) ** num_particles, 1)
    initial_state = State(rdd, shape, mesh, num_particles)

# Instatiating the walk
dtqw = DiscreteTimeQuantumWalk(coin, mesh, num_particles, phase=phase)

# Performing the walk
final_state = dtqw.walk(steps, initial_state)

# Measuring the state of the system and plotting its PDF
joint, collision, marginal = final_state.measure()
joint.plot(sim_path + 'joint_1d2p', dpi=300)
joint.plot_contour(sim_path + 'joint_1d2p_contour', dpi=300)
collision.plot(sim_path + 'collision_1d2p', dpi=300)

for p in range(len(marginal)):
    marginal[p].plot('{}marginal{}_1d2p'.format(sim_path, p + 1), dpi=300)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
collision.destroy()
for p in range(len(marginal)):
    marginal[p].destroy()
sparkContext.stop()
