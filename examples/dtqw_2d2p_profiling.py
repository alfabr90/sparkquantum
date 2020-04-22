import math
import cmath
import logging

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin2d.hadamard2d import Hadamard2D
from sparkquantum.dtqw.gauge.position_gauge import PositionGauge
from sparkquantum.dtqw.interaction.collision_phase_interaction import CollisionPhaseInteraction
from sparkquantum.dtqw.mesh.mesh2d.diagonal.lattice import LatticeDiagonal
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.qw_profiler import QuantumWalkProfiler
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.utils.utils import Utils

'''
    DTQW 2D - 2 particles
'''
base_path = './output/'
num_cores = 4
profile = True

num_particles = 2
steps = 5
size = 5
entangled = True
phase = 1.0 * cmath.pi

# Choosing a directory to store plots and logs
walk_path = "{}/{}_{}_{}_{}_{}_{}/".format(
    base_path, 'DiagonalLattice', 2 * size +
    1, steps, num_particles, phase, 'entangled' if entangled else 'not entangled'
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
coin = Hadamard2D()
mesh = LatticeDiagonal((size, size))

# Adding the profiler to the classes and starting it
profiler = QuantumWalkProfiler()

coin.profiler = profiler
mesh.profiler = profiler

coin_size = coin.size
mesh_size = mesh.size[0] * mesh.size[1]

# Options of initial states
if not entangled:
    # Center of the mesh
    positions = (int((mesh.size[0] - 1) / 2) * mesh.size[1] + int((mesh.size[1] - 1) / 2),
                 int((mesh.size[0] - 1) / 2) * mesh.size[1] + int((mesh.size[1] - 1) / 2))

    amplitudes = []

    # |i,j>|x,y> --> (|0,0>|0,0> + i|0,1>|0,0> - i|1,0>|0,0> + |1,1>|0,0>) / 2
    amplitudes.append((((1.0 + 0.0j) / 2),
                       ((0.0 + 1.0j) / 2),
                       ((0.0 - 1.0j) / 2),
                       ((1.0 + 0.0j) / 2)))

    # |i,j>|x,y> --> (|0,0>|0,0> + i|0,1>|0,0> + i|1,0>|0,0> - |1,1>|0,0>) / 2
    # amplitudes.append((((1.0 + 0.0j) / 2),
    #               ((0.0 + 1.0j) / 2),
    #               ((0.0 + 1.0j) / 2),
    #               ((-1.0 - 0.0j) / 2)))

    # |i,j>|x,y> --> (|0,0>|0,0> - |0,1>|0,0> - |1,0>|0,0> + |1,1>|0,0>) / 2
    # amplitudes.append((((1.0 + 0.0j) / 2),
    #               ((-1.0 - 0.0j) / 2),
    #               ((-1.0 - 0.0j) / 2),
    #               ((1.0 + 0.0j) / 2)))

    # |i,j>|x,y> --> (|0,0>|0,0> + i|0,1>|0,0> - i|1,0>|0,0> + |1,1>|0,0>) / 2
    amplitudes.append((((1.0 + 0.0j) / 2),
                       ((0.0 + 1.0j) / 2),
                       ((0.0 - 1.0j) / 2),
                       ((1.0 + 0.0j) / 2)))

    # |i,j>|x,y> --> (|0,0>|0,0> + i|0,1>|0,0> + i|1,0>|0,0> - |1,1>|0,0>) / 2
    # amplitudes.append((((1.0 + 0.0j) / 2),
    #               ((0.0 + 1.0j) / 2),
    #               ((0.0 + 1.0j) / 2),
    #               ((-1.0 - 0.0j) / 2)))

    # |i,j>|x,y> --> (|0,0>|0,0> - |0,1>|0,0> - |1,0>|0,0> + |1,1>|0,0>) / 2
    # amplitudes.append((((1.0 + 0.0j) / 2),
    #               ((-1.0 - 0.0j) / 2),
    #               ((-1.0 - 0.0j) / 2),
    #               ((1.0 + 0.0j) / 2)))

    initial_state = State.create(
        coin,
        mesh,
        positions,
        amplitudes,
        representationFormat)
else:
    # Center of the mesh
    position = int((mesh.size[0] - 1) / 2) * \
        mesh.size[1] + int((mesh.size[1] - 1) / 2)

    if representationFormat == Utils.StateRepresentationFormatCoinPosition:
        # |i1,j1>|x1,y1>|i2,j2>|x2,y2> --> (|1,1>|0,0>|0,0>|0,0> - |0,0>|0,0>|1,1>|0,0>) / sqrt(2)
        state = (
            ((3 * mesh_size + position) * coin_size * mesh_size +
             (0 * mesh_size + position), 1.0 / math.sqrt(2)),
            ((0 * mesh_size + position) * coin_size * mesh_size +
             (3 * mesh_size + position), -1.0 / math.sqrt(2))
        )
    elif representationFormat == Utils.StateRepresentationFormatPositionCoin:
        # |x1,y1>|i1,j1>|x2,y2>|i2,j2> --> (|0,0>|1,1>|0,0>|0,0> - |0,0>|0,0>|0,0>|1,1>) / sqrt(2)
        state = (
            ((position * coin_size + 3) * mesh_size * coin_size +
             (position * coin_size + 0), 1.0 / math.sqrt(2)),
            ((position * coin_size + 0) * mesh_size * coin_size +
             (position * coin_size + 3), -1.0 / math.sqrt(2))
        )

    rdd = sparkContext.parallelize(state)
    shape = ((coin_size * mesh_size) ** num_particles, 1)
    initial_state = State(rdd, shape, mesh, num_particles)

interaction = CollisionPhaseInteraction(num_particles, mesh, phase)

# Instatiating the walk
dtqw = DiscreteTimeQuantumWalk(
    coin,
    mesh,
    num_particles,
    interaction=interaction)

dtqw.profiler = profiler

# Performing the walk
final_state = dtqw.walk(steps, initial_state)

final_state.profiler = profiler

# Measuring the state of the system and plotting its PDF
gauge = PositionGauge()

gauge.profiler = profiler

joint, collision, marginal = gauge.measure(final_state)
collision.plot(walk_path + 'collision_2d2p', dpi=300)
collision.plot_contour(walk_path + 'collision_2d2p_contour', dpi=300)
for p in range(len(marginal)):
    marginal[p].plot('{}marginal{}_2d2p'.format(walk_path, p + 1), dpi=300)
    marginal[p].plot_contour(
        '{}marginal{}_2d2p_contour'.format(
            walk_path, p + 1), dpi=300)

# Exporting the profiling data
profiler.export(walk_path)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
collision.destroy()
for p in range(len(marginal)):
    marginal[p].destroy()
sparkContext.stop()
