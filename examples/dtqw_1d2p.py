import math
import cmath

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin1d.hadamard1d import Hadamard1D
from sparkquantum.dtqw.mesh.mesh1d.line import Line
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
profile = True

num_particles = 2
steps = 30
size = 30
entangled = True
phase = 1.0 * cmath.pi

representationFormat = Utils.RepresentationFormatCoinPosition
# representationFormat = Utils.RepresentationFormatPositionCoin

# Initiallizing the SparkContext
sparkConf = SparkConf().set('quantum.cluster.totalCores', num_cores)
sparkConf = sparkConf.set('quantum.representationFormat', representationFormat)
sparkContext = SparkContext(conf=sparkConf)
sparkContext.setLogLevel('ERROR')

# Choosing a coin and a mesh for the walk
coin = Hadamard1D(sparkContext)
mesh = Line(sparkContext, size)

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

# Adding the profiler to the classes and starting it
profiler = QuantumWalkProfiler()

coin.logger = Logger(coin.to_string(), sim_path)
mesh.logger = Logger(mesh.to_string(), sim_path)
coin.profiler = profiler
mesh.profiler = profiler

profiler.logger = Logger(profiler.to_string(), sim_path)
profiler.start()

coin_size = coin.size
mesh_size = mesh.size

# Center of the mesh
position = int((mesh_size - 1) / 2)

# Options of initial states
if not entangled:
    if representationFormat == Utils.RepresentationFormatCoinPosition:
        # |i>|x> --> (|0>|0> - i|1>|0>) / sqrt(2)
        state1 = (
            (0 * mesh_size + position, (1.0 + 0.0j) / math.sqrt(2)),
            (1 * mesh_size + position, (0.0 - 1.0j) / math.sqrt(2))
        )
        # |i>|x> --> |0>|0>
        # state1 = ((0 * mesh_size + position, (1.0 + 0.0j)), )
        # |i>|x> --> |1>|0>
        # state1 = ((1 * mesh_size + position, (1.0 + 0.0j)), )

        # |i>|x> --> (|0>|0> - i|1>|0>) / sqrt(2)
        state2 = (
            (0 * mesh_size + position, (1.0 + 0.0j) / math.sqrt(2)),
            (1 * mesh_size + position, (0.0 - 1.0j) / math.sqrt(2))
        )
        # |i>|x> --> |0>|0>
        # state2 = ((0 * mesh_size + position, (1.0 + 0.0j)), )
        # |i>|x> --> |1>|0>
        # state2 = ((1 * mesh_size + position, (1.0 + 0.0j)), )
    elif representationFormat == Utils.RepresentationFormatPositionCoin:
        # |x>|i> --> (|0>|0> - i|0>|1>) / sqrt(2)
        state1 = (
            (position * coin_size + 0, (1.0 + 0.0j) / math.sqrt(2)),
            (position * coin_size + 1, (0.0 - 1.0j) / math.sqrt(2))
        )
        # |x>|i> --> |0>|0>
        # state1 = ((position * coin_size + 0, (1.0 + 0.0j)), )
        # |x>|i> --> |0>|1>
        # state1 = ((position * coin_size + 1, (1.0 + 0.0j)), )

        # |x>|i> --> (|0>|0> - i|0>|1>) / sqrt(2)
        state2 = (
            (position * coin_size + 0, (1.0 + 0.0j) / math.sqrt(2)),
            (position * coin_size + 1, (0.0 - 1.0j) / math.sqrt(2))
        )
        # |x>|i> --> |0>|0>
        # state2 = ((position * coin_size + 0, (1.0 + 0.0j)), )
        # |x>|i> --> |0>|1>
        # state2 = ((position * coin_size + 1, (1.0 + 0.0j)), )

    shape = (coin_size * mesh_size, 1)

    base_state1 = State(sparkContext.parallelize(state1), shape, mesh, num_particles)
    base_state2 = State(sparkContext.parallelize(state2), shape, mesh, num_particles)

    initial_state = base_state1.kron(base_state2)

    base_state1.destroy()
    base_state2.destroy()
else:
    if representationFormat == Utils.RepresentationFormatCoinPosition:
        # |i1>|x1>|i2>|x2> --> (|1>|0>|0>|0> - |0>|0>|1>|0>) / sqrt(2)
        state = (
            ((1 * mesh_size + position) * coin_size * mesh_size + 0 * mesh_size + position, 1.0 / math.sqrt(2)),
            ((0 * mesh_size + position) * coin_size * mesh_size + 1 * mesh_size + position, -1.0 / math.sqrt(2)),
        )
    elif representationFormat == Utils.RepresentationFormatPositionCoin:
        # |x1>|i1>|x2>|i2> --> (|0>|1>|0>|0> - |0>|0>|0>|1>) / sqrt(2)
        state = (
            ((position * coin_size + 1) * mesh_size * coin_size + position * coin_size + 0, 1.0 / math.sqrt(2)),
            ((position * coin_size + 0) * mesh_size * coin_size + position * coin_size + 1, -1.0 / math.sqrt(2)),
        )

    rdd = sparkContext.parallelize(state)
    shape = ((coin_size * mesh_size) ** num_particles, 1)
    initial_state = State(rdd, shape, mesh, num_particles)

# Instatiating the walk
dtqw = DiscreteTimeQuantumWalk(sparkContext, coin, mesh, num_particles, phase=phase)

dtqw.logger = Logger(dtqw.to_string(), sim_path)
dtqw.profiler = profiler

# Performing the walk
final_state = dtqw.walk(steps, initial_state)

final_state.logger = Logger(final_state.to_string(), sim_path)
final_state.profiler = profiler

# Measuring the state of the system and plotting its PDF
joint, collision, marginal = final_state.measure()
joint.plot(sim_path + 'joint_1d2p', dpi=300)
joint.plot_contour(sim_path + 'joint_1d2p_contour', dpi=300)
collision.plot(sim_path + 'collision_1d2p', dpi=300)

for p in range(len(marginal)):
    marginal[p].plot('{}marginal{}_1d2p'.format(sim_path, p + 1), dpi=300)

# Exporting the profiling data
profiler.export(sim_path)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
collision.destroy()
for p in range(len(marginal)):
    marginal[p].destroy()
sparkContext.stop()
