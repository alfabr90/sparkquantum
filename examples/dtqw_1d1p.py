import math

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin1d.hadamard1d import Hadamard1D
from sparkquantum.dtqw.mesh.mesh1d.line import Line
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.qw_profiler import QuantumWalkProfiler
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.utils.utils import Utils
from sparkquantum.utils.logger import Logger

'''
    DTQW 1D - 1 particle
'''
base_path = './output/'
num_cores = 4
profile = True

num_particles = 1
steps = 30
size = 30

representationFormat = Utils.RepresentationFormatCoinPosition
# representationFormat = Utils.RepresentationFormatPositionCoin

# Initiallizing the SparkContext with some options
sparkConf = SparkConf().set('quantum.cluster.totalCores', num_cores)
sparkConf = sparkConf.set('quantum.representationFormat', representationFormat)
sparkContext = SparkContext(conf=sparkConf)
sparkContext.setLogLevel('ERROR')

# Choosing a coin and a mesh for the walk
coin = Hadamard1D(sparkContext)
mesh = Line(sparkContext, size)

# Choosing a directory to store plots and logs
walk_path = "{}/".format(
    base_path + Utils.filename(
        mesh.filename(), steps, num_particles
    )
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
if representationFormat == Utils.RepresentationFormatCoinPosition:
    # |i>|x> --> (|0>|0> - i|1>|0>) / sqrt(2)
    state = (
        (0 * mesh_size + position, (1.0 + 0.0j) / math.sqrt(2)),
        (1 * mesh_size + position, (0.0 - 1.0j) / math.sqrt(2))
    )
    # |i>|x> --> |0>|0>
    # state = ((0 * mesh_size + position, (1.0 + 0.0j)), )
    # |i>|x> --> |1>|0>
    # state = ((1 * mesh_size + position, (1.0 + 0.0j)), )
elif representationFormat == Utils.RepresentationFormatPositionCoin:
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
dtqw = DiscreteTimeQuantumWalk(sparkContext, coin, mesh, num_particles)

dtqw.logger = Logger(dtqw.to_string(), sim_path)
dtqw.profiler = profiler

# Performing the walk
final_state = dtqw.walk(steps, initial_state)

final_state.logger = Logger(final_state.to_string(), sim_path)
final_state.profiler = profiler

# Measuring the state of the system and plotting its PDF
joint = final_state.measure()
joint.plot(sim_path + 'joint_1d1p', dpi=300)

# Exporting the profiling data
profiler.export(sim_path)

# Destroying the RDD and stopping the SparkContext
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()
joint.destroy()
sparkContext.stop()
