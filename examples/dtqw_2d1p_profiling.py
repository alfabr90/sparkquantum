import math

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin2d.hadamard2d import Hadamard2D
from sparkquantum.dtqw.mesh.mesh2d.diagonal.lattice import LatticeDiagonal
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.qw_profiler import QuantumWalkProfiler
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.utils.utils import Utils
from sparkquantum.utils.logger import Logger

'''
    DTQW 2D - 1 particle
'''
base_path = './output/'
num_cores = 4
profile = True

num_particles = 1
steps = 30
size = 30

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
    'quantum.profiling.enabled', 'True'
)
sparkContext = SparkContext(conf=sparkConf)
sparkContext.setLogLevel('ERROR')

# Choosing a coin and a mesh for the walk
coin = Hadamard2D()
mesh = LatticeDiagonal((size, size))

# Adding a directory to store plots and logs
walk_path = "{}/".format(
    base_path + Utils.filename(
        mesh.filename(), steps, num_particles
    )
)

sim_path = walk_path
Utils.create_dir(sim_path)

# Adding the profiler to the classes and starting it
profiler = QuantumWalkProfiler()

coin.logger = Logger(coin.__class__.__name__, sim_path)
mesh.logger = Logger(mesh.__class__.__name__, sim_path)
coin.profiler = profiler
mesh.profiler = profiler

profiler.logger = Logger(profiler.__class__.__name__, sim_path)

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

dtqw.logger = Logger(dtqw.__class__.__name__, sim_path)
dtqw.profiler = profiler

# Performing the walk
final_state = dtqw.walk(steps, initial_state)

final_state.logger = Logger(final_state.__class__.__name__, sim_path)
final_state.profiler = profiler

# Measuring the state of the system and plotting its PDF
joint = final_state.measure()
joint.plot(sim_path + 'joint_2d1p', dpi=300)
joint.plot_contour(sim_path + 'joint_2d1p_contour', dpi=300)

# Exporting the profiling data
profiler.export(sim_path)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
sparkContext.stop()
