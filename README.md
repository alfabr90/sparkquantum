# sparkquantum

The idea of this project (perhaps it's still needed a better name for it) is to provide to the community a quantum algorithms simulator using [Apache Spark](https://spark.apache.org/). As they evolve, some quantum algorithms simulations acquire characteristics of a Big Data application due to the exponential grow suffered by their data structures. Thus, employing a framework like Apache Spark is a good approach, making it possible to execute larger simulations in high-performance computing (HPC) environments than when using single-processors, general purpose computers, allowing such data to be generated and processed at a reduced time in a parallel/distributed way. For now, the user can only simulate discrete time quantum walks (DTQW), but simulations of Grover's search and Shor's integer factorizarion algorithms are planned to be implemented soon.

## Requirements

To run the simulations, the [numpy](http://www.numpy.org/) package must be installed in the user's environment. Also, to plot the probability distribution functions, the user must install the [matplotlib](https://matplotlib.org/) package.

The _pyspark_ package is also required, but instead of installing it in the user's environment (which is an alternative), he/she may configure and launch Spark in such way that Python can access the package provided in Spark's directory.

## Discrete Time Quantum Walk

The `dtqw` module of the simulator allows the user to simulate one and two-dimensional DTQW composed by _n_ particles, with or without mesh percolations (broken links). By default, the following coins and meshes are already implemented:

- For one-dimensional walks:
  - Coin:
    - Hadamard
  - Mesh:
    - Line: a mesh based on the number of steps of the walk. To avoid the particles walking besides the boundaries of the mesh, its size is the double of the number of steps plus a center site where the particles **must** be located initially for a flawless simulation;
    - Segment: a line-based mesh with reflective sites on each border. The particles can start their walk at any site of the mesh;
    - Cycle: a line-based mesh with cyclic sites on each border. The particles can start their walk at any site of the mesh
- For two-dimensional walks:
  - Coin:
    - Hadamard
    - Grover
    - Fourier
  - Mesh:
    - Lattice: a mesh based on the number of steps of the walk. To avoid the particles walking besides the boundaries of the mesh, its size is the double of the number of steps plus a center site where the particles **must** be located initially for a flawless simulation. It's the Line's two-dimension counterpart;
    - Box: a lattice-based mesh with reflective sites on each coordinates' border. The particles can start their walk at any site of the mesh. It's the Segment's two-dimension counterpart;
    - Torus: a lattice-based mesh with cyclic sites on each coordinates' border. The particles can start their walk at any site of the mesh. It's the Cycle's two-dimension counterpart

TODO: explain how the user can implement custom coins and meshes

### Mesh Percolations

The user can simulate DTQWs with some mesh percolations. The already implemented variations are "permanent" and "random". For the former, the user must instantiate its corresponding class (`PermanentBrokenLinks`) passing in a list with the number of the edges that are broken. The second one is represented by the `RandomBrokenLinks` class, which needs to know the probability value that will be used to generate the broken edges of the mesh in a random fashion.

TODO: explain how the user can use permanent broken links and implement custom mesh percolations

### Example

Here, a simple example of how to simulate a DTQW with just one particle over a line mesh:

```python
import math

from pyspark import SparkContext, SparkConf

from sparkquantum.dtqw.coin.coin1d.hadamard import Hadamard
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.dtqw.gauge.position_gauge import PositionGauge
from sparkquantum.dtqw.mesh.mesh1d.line import Line
from sparkquantum.dtqw.state import State
from sparkquantum.utils.utils import Utils

# Initiallizing the SparkContext with some options
num_cores = 4

sparkConf = SparkConf().set('quantum.cluster.totalCores', num_cores)
sparkContext = SparkContext(conf=sparkConf)

# In this example, the walk will last 30 steps
# As we chose a `Line` mesh, its size will be
# automatically calculated, i.e., 2 * size + 1 sites
size = steps = 30

# Choosing a coin and a mesh for the walk
coin = Hadamard()
mesh = Line(size)

mesh_size = mesh.size

# Center of the mesh
# Notice that we set a tuple with only one element
# as we are simulating a DTQW with one particle
positions = [int((mesh_size - 1) / 2)]

# Options of initial states
# Notice that we set a tuple with only one element
# as we are simulating a DTQW with one particle
# |i>|x> --> (|0>|x> - i|1>|x>) / sqrt(2)
amplitudes = [[(1.0 + 0.0j) / math.sqrt(2),
               (0.0 - 1.0j) / math.sqrt(2)]]

# Building the initial state
initial_state = State.create(
                coin,
                mesh,
                positions,
                amplitudes)

# Instantiating the walk
dtqw = DiscreteTimeQuantumWalk(initial_state)

# Performing the walk
final_state = dtqw.walk(steps)

# Measuring the state of the system and plotting its PDF
gauge = PositionGauge()

joint = gauge.measure(final_state)
joint.plot('./joint_1d1p')

# Destroying the RDD and stopping the SparkContext
joint.destroy()
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()

sparkContext.stop()

```

For more detailed examples (e.g., of how to use profiling and mesh percolations), the user may refer to the files in the `examples` directory.

## Configuration Parameters

Below, there is a list of the current configuration parameters that the user can define accodingly to his/her needs. The values must be set through `SparkConf` key-value pairs:

| Property Name                                |                     Default                      | Meaning                                                                                                                                                          |
| :------------------------------------------- | :----------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| quantum.cluster.maxPartitionSize             |                       64MB                       | Maximum partition size of each RDD produced by the application. Only integers are accepted.                                                                      |
| quantum.cluster.numPartitionsSafetyFactor    |                       1.3                        | A safety factor when calculating the best possible number of partitions of each RDD produced by the application.                                                 |
| quantum.cluster.totalCores                   |                        1                         | Total number of cores used by the application. Necessary to calculate the best possible number of partitions of each RDD produced by the application             |
| quantum.cluster.useSparkDefaultNumPartitions |                      False                       | Whether to use the default number of partitions defined by Spark when partitioning the RDD produced by the application.                                          |
| quantum.dtqw.interactionOperator.checkpoint  |                      False                       | Whether to checkpoint the interaction operator. Considered only on interacting multiparticle walks.                                                              |
| quantum.dtqw.mesh.brokenLinks.generationMode |    `Utils.BrokenLinksGenerationModeBroadcast`    | The broken links generation mode. For now, can be as a broadcast variable or a RDD, both ways containing the edges numbers that are broken.                      |
| quantum.dtqw.profiler.logExecutors           |                      False                       | Whether to log executors' data if a profiler was provided.                                                                                                       |
| quantum.dtqw.state.dumpingFormat             |         `Utils.StateDumpingFormatIndex`          | Whether the system state has each of its elements dumped as vector indexes followed by their values or as mesh/cartesian coordinates followed by their values.   |
| quantum.dtqw.state.representationFormat      |  `Utils.StateRepresentationFormatCoinPosition`   | Whether the system state is represented by a kronecker product between the coin and position spaces or between the position and coin spaces.                     |
| quantum.dtqw.walk.checkpointingFrequency     |                        -1                        | The frequency to checkpoint the states. A state will be checkpointed at every _n_ steps. When -1, it never checkpoints the state.                                |
| quantum.dtqw.walk.checkUnitary               |                      False                       | Whether to check if each state is unitary.                                                                                                                       |
| quantum.dtqw.walk.dumpingFrequency           |                        -1                        | The frequency of dumping the states to disk. A state will be dumped at every _n_ steps. When -1, it never dumps the state.                                       |
| quantum.dtqw.walk.dumpingPath                |                       "./"                       | The directory to save the dumps.                                                                                                                                 |
| quantum.dtqw.walk.dumpStatesPDF              |                      False                       | Whether to dump to disk the PDF of each state.                                                                                                                   |
| quantum.dtqw.walkOperator.checkpoint         |                      False                       | Whether to checkpoint the walk operator(s).                                                                                                                      |
| quantum.dtqw.walkOperator.kroneckerMode      |          `Utils.KroneckerModeBroadcast`          | The kronecker product mode to build the walk operator(s). For now, can be as a broadcast variable or a RDD.                                                      |
| quantum.dtqw.walkOperator.tempPath           |                       "./"                       | The temporary directory to save the walk operators' dump. Considered only when the kronecker mode is `Utils.KroneckerModeDump`.                                  |
| quantum.dumpingCompressionCodec              |                       None                       | Compression codec class used by Spark when dumping each RDD's data disk.                                                                                         |
| quantum.dumpingGlue                          |                       " "                        | A string to connect the coordinates of each RDD's element when dumping its data.                                                                                 |
| quantum.logging.enabled                      |                      False                       | Whether the application must use [Python's logging facility](https://docs.python.org/3/library/logging.html).                                                    |
| quantum.logging.filename                     |                   './log.txt'                    | The filename (with relative or absolute path) where all log data will be written by Python's logging facility.                                                   |
| quantum.logging.format                       | '%(levelname)s:%(name)s:%(asctime)s:%(message)s' | Python's logging.Formatter acceptable format which each log record will be at.                                                                                   |
| quantum.logging.level                        |                `logging.WARNING`                 | Python's logging facility acceptable severity level.                                                                                                             |
| quantum.math.dumpingMode                     |           `Utils.DumpingModePartFiles`           | Whether the mathematical entity's (Matrix, Vector, Operator, State, etc.) RDD has its data dumped to disk in a single file or in multiple part-\* files in disk. |
| quantum.math.roundPrecision                  |                        10                        | Decimal precision when rounding numbers.                                                                                                                         |
| quantum.profiling.enabled                    |                      False                       | Whether the application must profile all created RDD to get their metrics.                                                                                       |
| quantum.profiling.baseUrl                    |         'http://localhost:4040/api/v1/'          | The [Spark Rest API](http://spark.apache.org/docs/latest/monitoring.html#rest-api) base URL that the application's profiler must use to get some metrics.        |
