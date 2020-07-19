import math
import cmath

import numpy as np
from pyspark import SparkContext, SparkConf

from sparkquantum import constants, plot, util
from sparkquantum.dtqw.coin.coin1d.hadamard import Hadamard
from sparkquantum.dtqw.gauge.position_gauge import PositionGauge
from sparkquantum.dtqw.mesh.mesh1d.line import Line
from sparkquantum.dtqw.interaction.collision_phase_interaction import CollisionPhaseInteraction
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.profiler import QuantumWalkProfiler
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk

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

# Choosing a directory to store plots and logs
walk_path = "{}/{}_{}_{}_{}_{}_{}/".format(
    base_path, 'Line', 2 * size +
    1, steps, num_particles, phase, 'entangled' if entangled else 'not entangled'
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

# Choosing a coin and a mesh for the walk
coin = Hadamard()
mesh = Line(size)

coin_size = coin.size
mesh_size = mesh.size

interaction = None  # CollisionPhaseInteraction(num_particles, mesh, phase)

# Options of initial states
if not entangled:
    # Center of the mesh
    positions = [mesh.center(), mesh.center()]

    amplitudes = []

    # |i>|x> --> (|0>|x> - i|1>|x>) / sqrt(2)
    amplitudes.append([(1.0 + 0.0j) / math.sqrt(2),
                       (0.0 - 1.0j) / math.sqrt(2)])

    # |i>|x> --> |0>|x>
    # amplitudes.append([(1.0 + 0.0j), 0])

    # |i>|x> --> |1>|x>
    # amplitudes.append([0, (1.0 + 0.0j)])

    # |i>|x> --> (|0>|x> - i|1>|x>) / sqrt(2)
    amplitudes.append([(1.0 + 0.0j) / math.sqrt(2),
                       (0.0 - 1.0j) / math.sqrt(2)])

    # |i>|x> --> |0>|x>
    # amplitudes.append([(1.0 + 0.0j), 0])

    # |i>|x> --> |1>|x>
    # amplitudes.append([0, (1.0 + 0.0j)])

    initial_state = State.create(
        coin,
        mesh,
        positions,
        amplitudes,
        interaction=interaction,
        representationFormat=representationFormat)
else:
    # Center of the mesh
    position = mesh.center()

    if representationFormat == constants.StateRepresentationFormatCoinPosition:
        # |i1>|x1>|i2>|x2> --> (|1>|x1>|0>|x2> - |0>|x1>|1>|x2>) / sqrt(2)
        state = [[(1 * mesh_size + position) * coin_size * mesh_size + 0 * mesh_size + position, 1, 1.0 / math.sqrt(2)],
                 [(0 * mesh_size + position) * coin_size * mesh_size + 1 * mesh_size + position, 1, -1.0 / math.sqrt(2)]]

        # |i1>|x1>|i2>|x2> --> (|1>|x1>|0>|x2> + |0>|x1>|1>|x2>) / sqrt(2)
        # state = [[(1 * mesh_size + position) * coin_size * mesh_size + 0 * mesh_size + position, 1, 1.0 / math.sqrt(2)],
        #          [(0 * mesh_size + position) * coin_size * mesh_size + 1 * mesh_size + position, 1, 1.0 / math.sqrt(2)]]
    elif representationFormat == constants.StateRepresentationFormatPositionCoin:
        # |x1>|i1>|x2>|i2> --> (|x1>|1>|x2>|0> - |x1>|0>|x2>|1>) / sqrt(2)
        state = [[(position * coin_size + 1) * mesh_size * coin_size + position * coin_size + 0, 1, 1.0 / math.sqrt(2)],
                 [(position * coin_size + 0) * mesh_size * coin_size + position * coin_size + 1, 1, -1.0 / math.sqrt(2)]]

        # |x1>|i1>|x2>|i2> --> (|x1>|1>|x2>|0> + |x1>|0>|x2>|1>) / sqrt(2)
        # state = [[(position * coin_size + 1) * mesh_size * coin_size + position * coin_size + 0, 1, 1.0 / math.sqrt(2)],
        #          [(position * coin_size + 0) * mesh_size * coin_size + position * coin_size + 1, 1, 1.0 / math.sqrt(2)]]

    rdd = sparkContext.parallelize(state)
    shape = [(coin_size * mesh_size) ** num_particles, 1]
    initial_state = State(
        rdd,
        shape,
        coin,
        mesh,
        num_particles,
        interaction=interaction)

# Instantiating the walk
dtqw = DiscreteTimeQuantumWalk(initial_state)

# Performing the walk
final_state = dtqw.walk(steps)

# Measuring the state of the system and plotting its probability distribution
gauge = PositionGauge()

joint, collision, marginal = gauge.measure(final_state)

axis = np.meshgrid(mesh.axis(), mesh.axis(), indexing='ij')
data = joint.ndarray()
labels = [v.name for v in joint.variables]

plot.surface(axis, data, walk_path + 'joint_1d2p',
             labels=labels + ['Probability'], dpi=300)
plot.contour(axis, data, walk_path + 'joint_1d2p_contour',
             labels=labels, dpi=300)

axis = mesh.axis()
data = collision.ndarray()
labels = [v.name for v in collision.variables] + ['Probability']

plot.line(axis, data, walk_path + 'collision_1d2p', labels=labels, dpi=300)

for p in range(len(marginal)):
    plot.line(axis, marginal[p].ndarray(), '{}marginal{}_1d2p'.format(walk_path, p + 1),
              labels=[v.name for v in marginal[p].variables] + ['Probability'], dpi=300)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
collision.destroy()
for p in range(len(marginal)):
    marginal[p].destroy()
sparkContext.stop()
