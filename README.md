# sparkquantum

The idea of this project (it's still needed a better name for it) is to provide to the community a quantum algorithms simulator using [Apache Spark](https://spark.apache.org/). For now, the user can only simulate discrete time quantum walks, but simulations of Grover's search and Shor's integer factorizarion algorithms are planned to be implemented soon.

## Requirements

To run the simulations, the [numpy](http://www.numpy.org/) package must be installed in the user's environment. Also, to plot the probability distribution functions, the user must install the [matplotlib](https://matplotlib.org/) package.

The *pyspark* package is also required, but instead of installing it in the user's environment (which is an alternative), he/she may configure and launch Spark in such way that Python can access the package provided in Spark's directory.

## Discrete Time Quantum Walk

The `dtqw` module of the simulator allows the user to simulate one and two-dimensional discrete time quantum walks composed by *n* particles. The supported coins and meshes in those cases are:

* For one-dimensional walks:
    * Coin:
        * Hadamard
    * Mesh:
        * Line: a mesh based on the number of steps of the walk. To avoid the particles walking besides the boundaries of the mesh, its size is the double of the number of steps plus a center site where the particles **must** be located initially for a correct result;
        * Segment: a line-based mesh with reflective sites on each border. The particles can start their walk at any site of the mesh;
        * Cycle: a line-based mesh with cyclic sites on each border. The particles can start their walk at any site of the mesh
* For two-dimensional walks:
    * Coin:
        * Hadamard
        * Grover
        * Fourier
    * Mesh:
        * Lattice: a mesh based on the number of steps of the walk. To avoid the particles walking besides the boundaries of the mesh, its size is the double of the number of steps plus a center site where the particles **must** be located initially for a correct result. It's the Line's two-dimension counterpart;
        * Box: a lattice-based mesh with reflective sites on each coordinates' border. The particles can start their walk at any site of the mesh. It's the Segment's two-dimension counterpart;
        * Torus: a lattice-based mesh with cyclic sites on each coordinates' border. The particles can start their walk at any site of the mesh. It's the Cycle's two-dimension counterpart

### Example

Here, a simple example of how to simulate a discrete time quantum walk with just one particle over a line mesh:

```python
import math

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin1d.hadamard1d import Hadamard1D
from sparkquantum.dtqw.mesh.mesh1d.line import Line
from sparkquantum.dtqw.state import State
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.utils.utils import Utils

# Initiallizing the SparkContext with some options
num_cores = 4

sparkConf = SparkConf().set('quantum.cluster.totalCores', num_cores)
sparkContext = SparkContext(conf=sparkConf)

# In this example, the walk will last 30 steps
size = steps = 30

# Choosing a coin and a mesh for the walk
coin = Hadamard1D(sparkContext)
mesh = Line(sparkContext, size)

coin_size = coin.size
mesh_size = mesh.size

# Center of the mesh
position = int((mesh_size - 1) / 2)

# |psi> = |coin>|position> --> (|0>|0> - i|1>|0>) / sqrt(2)
state = (
    (0 * mesh_size + position, (1.0 + 0.0j) / math.sqrt(2)),
    (1 * mesh_size + position, (0.0 - 1.0j) / math.sqrt(2))
)

# Building the initial state
rdd = sparkContext.parallelize(state)
shape = (coin_size * mesh_size, 1)
initial_state = State(rdd, shape, mesh, num_particles)

# Instatiating the walk
dtqw = DiscreteTimeQuantumWalk(sparkContext, coin, mesh, num_particles)

# Performing the walk
final_state = dtqw.walk(steps, initial_state)

# Measuring the state of the system and plotting its PDF
joint = final_state.measure()
joint.plot('./joint_1d1p')

# Destroying the RDD and stopping the SparkContext
joint.destroy()
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()

sparkContext.stop()

```

For more detailed examples, the user may refer to the files in the `examples` directory.

## Configuration Parameters

Below, there is a list of the current configuration parameters that the user can define accodingly to his/her needs. The values must be set through `SparkConf` key-value pairs:

| Property Name | Default | Meaning |
| :--- | :---: | :--- |
| quantum.dumpingGlue | " " | A string to connect the coordinates of each RDD's element when dumping its data. |
| quantum.dumpingCompressionCodec | None | Compression coded class used by Spark when saving data to disk. |
| quantum.math.roundPrecision | 10 | Decimal precision when rounding numbers. |
| quantum.cluster.numPartitionsSafetyFactor | 1.3 | A safety factor when calculating the best possible number of partitions of each RDD produced by the application. |
| quantum.cluster.useSparkDefaultNumPartitions | False | Whether to use the default number of partitions defined by Spark when partitioning the RDD. |
| quantum.cluster.totalCores | 1 | Total number of cores used by the application. |
| quantum.cluster.maxPartitionSize | 64MB | Maximum partition size of each RDD produced by the application. |
| quantum.dtqw.mesh.brokenLinks.generationMode | `Utils.BrokenLinksGenerationModeBroadcast` | The broken links generation mode. For now, can be as a broadcast variable or a RDD, both ways containing the edges numbers that are broken. |
| quantum.dtqw.mesh.randomBrokenLinks.seed | `time.time()` | Seed for randomization of the broken links throughout the mesh. |
| quantum.dtqw.interactionOperator.checkpoint | False | Whether to checkpoint the interaction operator. Considered only on interacting multiparticle walks. |
| quantum.dtqw.walkOperator.checkpoint | False | Whether to checkpoint the walk operator(s). |
| quantum.dtqw.walkOperator.kroneckerMode | `Utils.KroneckerModeBroadcast` | The kronecker product mode to build the walk operator(s). For now, can be as a broadcast variable or a RDD. |
| quantum.dtqw.walkOperator.tempPath | "./" | The temporary directory to save the walk operators' dump. Considered only when the kronecker mode is `Utils.KroneckerModeDump`. |
| quantum.dtqw.walk.checkpointStates | False | Whether to checkpoint the states produced by each step of the walk. |
| quantum.dtqw.walk.checkpointingFrequency | -1 | The frequency of checkpointing the states. A state will be checkpointed every *n* steps. Considered only when checkpointStates evaluates to True. -1 never checkpoints the state. |
| quantum.dtqw.walk.dumpStates | False | Whether to dump to disk the states produced by each step of the walk. |
| quantum.dtqw.walk.dumpingFrequency | -1 | The frequency of dumping the states to disk. A state will be dumped every *n* steps. Considered only when dumpStates evaluates to True. -1 never checkpoints the state. |
| quantum.dtqw.walk.dumpingPath | "./" | The directory to save the dumps. |
| quantum.dtqw.walk.checkUnitary | False | Whether to check if every state is unitary. |
| quantum.dtqw.state.representationFormat | `Utils.StateRepresentationFormatCoinPosition` | Whether the system state is represented by a kronecker product between the coin and position spaces or between the position and coin spaces. |
| quantum.dtqw.state.dumpingFormat | `Utils.StateDumpingFormatIndex` | Whether the system state has each of its elements dumped as vector indexes followed by their values or as mesh/cartesian coordinates followed by their values. |
| quantum.dtqw.profiler.logExecutors | False | Whether to log executors' data if a profiler was provided. |
