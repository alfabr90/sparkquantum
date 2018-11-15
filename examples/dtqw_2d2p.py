import math
import cmath

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin2d.hadamard2d import Hadamard2D
from sparkquantum.dtqw.mesh.mesh2d.diagonal.lattice import LatticeDiagonal
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.qw_profiler import QuantumWalkProfiler
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.utils.utils import Utils
from sparkquantum.utils.logger import Logger

'''
    DTQW2D - 2 particles
'''
base_path = './output/'
num_cores = 4
profile = True

num_particles = 2
steps = 3
size = 3
entangled = True
phase = 1.0 * cmath.pi

representationFormat = Utils.RepresentationFormatCoinPosition
# representationFormat = Utils.RepresentationFormatPositionCoin

# Initiallizing the SparkContext
sparkConf = SparkConf().set('quantum.cluster.totalCores', num_cores)
sparkContext = SparkContext(conf=sparkConf)
sparkContext.setLogLevel("ERROR")

# Choosing a coin and a mesh for the walk
coin = Hadamard2D(sparkContext)
mesh = LatticeDiagonal(sparkContext, (size, size))

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
mesh_size = mesh.size[0] * mesh.size[1]

# Center of the mesh
position = int((mesh.size[0] - 1) / 2) * mesh.size[1] + int((mesh.size[1] - 1) / 2)

# Options of initial states
if not entangled:
    if representationFormat == Utils.RepresentationFormatCoinPosition:
        # |i,j>|x,y> --> (|0,0>|0,0> + i|0,1>|0,0> - i|1,0>|0,0> + |1,1>|0,0>) / 2
        state1 = (
            (0 * mesh_size + position, (1.0 + 0.0j) / 2),
            (1 * mesh_size + position, (0.0 + 1.0j) / 2),
            (2 * mesh_size + position, (0.0 - 1.0j) / 2),
            (3 * mesh_size + position, (1.0 + 0.0j) / 2)
        )
        # |i,j>|x,y> --> (|0,0>|0,0> + i|0,1>|0,0> + i|1,0>|0,0> - |1,1>|0,0>) / 2
        '''state1 = (
            (0 * mesh_size + position, (1.0 + 0.0j) / 2),
            (1 * mesh_size + position, (0.0 + 1.0j) / 2),
            (2 * mesh_size + position, (0.0 + 1.0j) / 2),
            (3 * mesh_size + position, (-1.0 - 0.0j) / 2)
        )'''
        # |i,j>|x,y> --> (|0,0>|0,0> - |0,1>|0,0> - |1,0>|0,0> + |1,1>|0,0>) / 2
        '''state1 = (
            (0 * mesh_size + position, (1.0 + 0.0j) / 2),
            (1 * mesh_size + position, (-1.0 - 0.0j) / 2),
            (2 * mesh_size + position, (-1.0 - 0.0j) / 2),
            (3 * mesh_size + position, (1.0 + 0.0j) / 2)
        )'''

        # |i,j>|x,y> --> (|0,0>|0,0> + i|0,1>|0,0> - i|1,0>|0,0> + |1,1>|0,0>) / 2
        state2 = (
            (0 * mesh_size + position, (1.0 + 0.0j) / 2),
            (1 * mesh_size + position, (0.0 + 1.0j) / 2),
            (2 * mesh_size + position, (0.0 - 1.0j) / 2),
            (3 * mesh_size + position, (1.0 + 0.0j) / 2)
        )
        # |i,j>|x,y> --> (|0,0>|0,0> + i|0,1>|0,0> + i|1,0>|0,0> - |1,1>|0,0>) / 2
        '''state2 = (
            (0 * mesh_size + position, (1.0 + 0.0j) / 2),
            (1 * mesh_size + position, (0.0 + 1.0j) / 2),
            (2 * mesh_size + position, (0.0 + 1.0j) / 2),
            (3 * mesh_size + position, (-1.0 - 0.0j) / 2)
        )'''
        # |i,j>|x,y> --> (|0,0>|0,0> - |0,1>|0,0> - |1,0>|0,0> + |1,1>|0,0>) / 2
        '''state2 = (
            (0 * mesh_size + position, (1.0 + 0.0j) / 2),
            (1 * mesh_size + position, (-1.0 - 0.0j) / 2),
            (2 * mesh_size + position, (-1.0 - 0.0j) / 2),
            (3 * mesh_size + position, (1.0 + 0.0j) / 2)
        )'''
    elif representationFormat == Utils.RepresentationFormatPositionCoin:
        # |x,y>|i,j> --> (|0,0>|0,0> + i|0,0>|0,1> - i|0,0>|1,0> + |0,0>|1,1>) / 2
        state1 = (
            (position * coin_size + 0, (1.0 + 0.0j) / 2),
            (position * coin_size + 1, (0.0 + 1.0j) / 2),
            (position * coin_size + 2, (0.0 - 1.0j) / 2),
            (position * coin_size + 3, (1.0 + 0.0j) / 2)
        )
        # |x,y>|i,j> --> (|0,0>|0,0> + i|0,0>|0,1> + i|0,0>|1,0> - |0,0>|1,1>) / 2
        '''state1 = (
            (position * coin_size + 0, (1.0 + 0.0j) / 2),
            (position * coin_size + 1, (0.0 + 1.0j) / 2),
            (position * coin_size + 2, (0.0 + 1.0j) / 2),
            (position * coin_size + 3, (-1.0 - 0.0j) / 2)
        )'''
        # |x,y>|i,j> --> (|0,0>|0,0> - |0,0>|0,1> - |0,0>|1,0> + |0,0>|1,1>) / 2
        '''state1 = (
            (position * coin_size + 0, (1.0 + 0.0j) / 2),
            (position * coin_size + 1, (-1.0 - 0.0j) / 2),
            (position * coin_size + 2, (-1.0 - 0.0j) / 2),
            (position * coin_size + 3, (1.0 + 0.0j) / 2)
        )'''

        # |x,y>|i,j> --> (|0,0>|0,0> + i|0,0>|0,1> - i|0,0>|1,0> + |0,0>|1,1>) / 2
        state2 = (
            (position * coin_size + 0, (1.0 + 0.0j) / 2),
            (position * coin_size + 1, (0.0 + 1.0j) / 2),
            (position * coin_size + 2, (0.0 - 1.0j) / 2),
            (position * coin_size + 3, (1.0 + 0.0j) / 2)
        )
        # |x,y>|i,j> --> (|0,0>|0,0> + i|0,0>|0,1> + i|0,0>|1,0> - |0,0>|1,1>) / 2
        '''state2 = (
            (position * coin_size + 0, (1.0 + 0.0j) / 2),
            (position * coin_size + 1, (0.0 + 1.0j) / 2),
            (position * coin_size + 2, (0.0 + 1.0j) / 2),
            (position * coin_size + 3, (-1.0 - 0.0j) / 2)
        )'''
        # |x,y>|i,j> --> (|0,0>|0,0> - |0,0>|0,1> - |0,0>|1,0> + |0,0>|1,1>) / 2
        '''state2 = (
            (position * coin_size + 0, (1.0 + 0.0j) / 2),
            (position * coin_size + 1, (-1.0 - 0.0j) / 2),
            (position * coin_size + 2, (-1.0 - 0.0j) / 2),
            (position * coin_size + 3, (1.0 + 0.0j) / 2)
        )'''

    shape = (coin_size * mesh_size, 1)

    base_state1 = State(sparkContext.parallelize(state1), shape, mesh, num_particles)
    base_state2 = State(sparkContext.parallelize(state2), shape, mesh, num_particles)

    initial_state = base_state1.kron(base_state2)

    base_state1.destroy()
    base_state2.destroy()
else:
    if representationFormat == Utils.RepresentationFormatCoinPosition:
        # |i1,j1>|x1,y1>|i2,j2>|x2,y2> --> (|1,1>|0,0>|0,0>|0,0> - |0,0>|0,0>|1,1>|0,0>) / sqrt(2)
        state = (
            ((3 * mesh_size + position) * coin_size * mesh_size + (0 * mesh_size + position), 1.0 / math.sqrt(2)),
            ((0 * mesh_size + position) * coin_size * mesh_size + (3 * mesh_size + position), -1.0 / math.sqrt(2))
        )
    elif representationFormat == Utils.RepresentationFormatPositionCoin:
        # |x1,y1>|i1,j1>|x2,y2>|i2,j2> --> (|0,0>|1,1>|0,0>|0,0> - |0,0>|0,0>|0,0>|1,1>) / sqrt(2)
        state = (
            ((position * coin_size + 3) * mesh_size * coin_size + (position * coin_size + 0), 1.0 / math.sqrt(2)),
            ((position * coin_size + 0) * mesh_size * coin_size + (position * coin_size + 3), -1.0 / math.sqrt(2))
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
collision.plot(sim_path + 'collision_2d2p', dpi=300)
collision.plot_contour(sim_path + 'collision_2d2p_contour', dpi=300)
for p in range(len(marginal)):
    marginal[p].plot(sim_path + 'marginal{}_2d2p'.format(p + 1), dpi=300)
    marginal[p].plot_contour(sim_path + 'marginal{}_2d2p_contour'.format(p + 1), dpi=300)

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
