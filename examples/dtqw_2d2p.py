import math
import cmath

from pyspark import SparkContext, SparkConf

from sparkquantum import constants, util
from sparkquantum.dtqw.coin.coin2d.hadamard import Hadamard
from sparkquantum.dtqw.gauge.position_gauge import PositionGauge
from sparkquantum.dtqw.interaction.collision_phase_interaction import CollisionPhaseInteraction
from sparkquantum.dtqw.mesh.mesh2d.diagonal.lattice import Lattice
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.profiler import QuantumWalkProfiler
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk

'''
    DTQW 2D - 2 particles
'''
base_path = './output/'
num_cores = 4

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
mesh = Lattice((size, size))

coin_size = coin.size
mesh_size = mesh.size[0] * mesh.size[1]

interaction = CollisionPhaseInteraction(num_particles, mesh, phase)

# Options of initial states
if not entangled:
    # Center of the mesh
    positions = [mesh.center(), mesh.center()]

    amplitudes = []

    # |i,j>|x,y> --> (|0,0>|x,y> + i|0,1>|x,y> - i|1,0>|x,y> + |1,1>|x,y>) / 2
    amplitudes.append([(1.0 + 0.0j) / 2,
                       (0.0 + 1.0j) / 2,
                       (0.0 - 1.0j) / 2,
                       (1.0 + 0.0j) / 2])

    # |i,j>|x,y> --> (|0,0>|x,y> + i|0,1>|x,y> + i|1,0>|x,y> - |1,1>|x,y>) / 2
    # amplitudes.append([(1.0 + 0.0j) / 2,
    #                    (0.0 + 1.0j) / 2,
    #                    (0.0 + 1.0j) / 2,
    #                    (-1.0 - 0.0j) / 2])

    # |i,j>|x,y> --> (|0,0>|x,y> - |0,1>|x,y> - |1,0>|x,y> + |1,1>|x,y>) / 2
    # amplitudes.append([(1.0 + 0.0j) / 2,
    #                    (-1.0 - 0.0j) / 2,
    #                    (-1.0 - 0.0j) / 2,
    #                    (1.0 + 0.0j) / 2])

    # |i,j>|x,y> --> (|0,0>|x,y> + i|0,1>|x,y> - i|1,0>|x,y> + |1,1>|x,y>) / 2
    amplitudes.append([(1.0 + 0.0j) / 2,
                       (0.0 + 1.0j) / 2,
                       (0.0 - 1.0j) / 2,
                       (1.0 + 0.0j) / 2])

    # |i,j>|x,y> --> (|0,0>|x,y> + i|0,1>|x,y> + i|1,0>|x,y> - |1,1>|x,y>) / 2
    # amplitudes.append([(1.0 + 0.0j) / 2,
    #                    (0.0 + 1.0j) / 2,
    #                    (0.0 + 1.0j) / 2,
    #                    (-1.0 - 0.0j) / 2])

    # |i,j>|x,y> --> (|0,0>|x,y> - |0,1>|x,y> - |1,0>|x,y> + |1,1>|x,y>) / 2
    # amplitudes.append([(1.0 + 0.0j) / 2,
    #                    (-1.0 - 0.0j) / 2,
    #                    (-1.0 - 0.0j) / 2,
    #                    (1.0 + 0.0j) / 2])

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
        # |i1,j1>|x1,y1>|i2,j2>|x2,y2> --> (|1,1>|x1,y1>|0,0>|x2,y2> - |0,0>|x1,y1>|1,1>|x2,y2>) / sqrt(2)
        state = [[(3 * mesh_size + position) * coin_size * mesh_size + (0 * mesh_size + position), 1, 1.0 / math.sqrt(2)],
                 [(0 * mesh_size + position) * coin_size * mesh_size + (3 * mesh_size + position), 1, -1.0 / math.sqrt(2)]]
    elif representationFormat == constants.StateRepresentationFormatPositionCoin:
        # |x1,y1>|i1,j1>|x2,y2>|i2,j2> --> (|x1,y1>|1,1>|x2,y2>|0,0> - |x1,y1>|0,0>|x2,y2>|1,1>) / sqrt(2)
        state = [[(position * coin_size + 3) * mesh_size * coin_size + (position * coin_size + 0), 1, 1.0 / math.sqrt(2)],
                 [(position * coin_size + 0) * mesh_size * coin_size + (position * coin_size + 3), 1, -1.0 / math.sqrt(2)]]

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
collision.plot(walk_path + 'collision_2d2p', dpi=300)
collision.plot_contour(walk_path + 'collision_2d2p_contour', dpi=300)
for p in range(len(marginal)):
    marginal[p].plot('{}marginal{}_2d2p'.format(walk_path, p + 1), dpi=300)
    marginal[p].plot_contour(
        '{}marginal{}_2d2p_contour'.format(
            walk_path, p + 1), dpi=300)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
collision.destroy()
for p in range(len(marginal)):
    marginal[p].destroy()

sparkContext.stop()
