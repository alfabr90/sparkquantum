# sparkquantum

The idea of this project (perhaps it's still needed a better name for it) is to provide to the community a quantum algorithms simulator written in [Python](https://www.python.org/) using [Apache Spark](https://spark.apache.org/).

As they evolve, some quantum algorithms simulations acquire characteristics of a Big Data application due to the exponential grow suffered by their data structures. Thus, employing a framework like Apache Spark is a good approach, making it possible to execute larger simulations in high-performance computing (HPC) environments than when using single-processors, general purpose computers, allowing such data to be generated and processed at a reduced time in a parallel/distributed way.

## Requirements

To run the simulations, the [numpy](http://www.numpy.org/) package must be installed in the user's environment. Also, to plot the probability distributions, the user must install the [matplotlib](https://matplotlib.org/) package.

The [pyspark](https://pypi.org/project/pyspark/) package is also required, but instead of installing it in the user's environment (which is an alternative), he/she may configure and launch Spark in such way that Python can access the package provided in Spark's directory.

## How to Use It

In contrast to other simulators, this one expects a programmatic way to input the initial conditions of the simulations, but its use is still easy and simple, requiring few steps to start using it.

For now, the user can only simulate discrete time quantum walks (DTQW), but simulations of Grover's search and Shor's integer factorizarion algorithms are planned to be implemented soon.

To check all source code documentation (Python docstrings), visit this project's page in [Read the Docs](https://sparkquantum.readthedocs.io/en/latest/).

### Discrete Time Quantum Walk

As DTQW is characterized as an iterative algorithm, this simulator considers each step as a matrix-vector multiplication - the matrix represents the unitary evolution operator and the vector represents the current state of the system.

The `dtqw` module of the simulator allows the user to simulate DTQW composed by _n_ particles over any kind of mesh (see the list of available meshes already implemented), with or without mesh percolations (broken links), and, in the end, measure the final quantum state so that a probability distribution of the particles' possible positions can be obtained and plotted.

#### A Simple Example

In order to simulate, for instance, a particle walking over a line, the user must, first of all, choose a coin:

```python
coin = Hadamard()
```

and a mesh:

```python
mesh = Line(steps)
```

Particularly for a `Line` mesh, the number of steps of the walk can be passed in as a parameter. This class is responsible to build a line mesh with size comprehending a central site where the particle **must**, initially, be located at, and the number of steps to the left and to the right of that central site. This avoids the particles walking besides the boundaries of the mesh.

Next, the user must provide the initial position of the particle:

```python
positions = [mesh.center()]
```

Remember that, for line meshes, its center **must** be the initial position of the particle. Besides, as the simulator allows DTQW with _n_ particles, the `positions` variable must be an array-like structure, similarly to the following `amplitudes` variable, containing the amplitudes of the initial quantum state:

```python
amplitudes = [[(1.0 + 0.0j) / math.sqrt(2),
               (0.0 - 1.0j) / math.sqrt(2)]]
```

In this example, the above amplitude and position values correspond to the initial quantum state, in Dirac's notation:

`|c>|p> --> (|0> - i|1>)|p> / sqrt(2) = (|0>|p> - i|1>|p>) / sqrt(2)`,

where `c` is the coin state and `p` is the position state.

Thus, the initial state can be built using the static `State.create` method with all the previous data:

```python
initial_state = State.create(coin, mesh, positions, amplitudes)
```

To perform the walk, the user must instantiate a `DiscreteTimeQuantumWalk` object with the recently built quantum state:

```python
dtqw = DiscreteTimeQuantumWalk(initial_state)
```

and call its `walk` method, informing the desired number of steps:

```python
final_state = dtqw.walk(steps)
```

Finally, the user can measure the final quantum state, obtaining the probability distribution of the possible particle's positions and plot it:

```python
gauge = PositionGauge()

joint = gauge.measure(final_state)

plot.line(mesh.axis(), joint.ndarray(), 'joint_1d1p',
          labels=['Position', 'Probability'])

```

Below, the complete Python script of the previous example with some Spark related commands:

```python
import math

from pyspark import SparkContext, SparkConf

from sparkquantum import plot, util
from sparkquantum.dtqw.coin.coin1d.hadamard import Hadamard
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.dtqw.gauge.position import PositionGauge
from sparkquantum.dtqw.mesh.mesh1d.line import Line
from sparkquantum.dtqw.state import State

# Initiallizing the SparkContext with some options
num_cores = 4

sparkConf = SparkConf().set('sparkquantum.cluster.totalCores', num_cores)
sparkContext = SparkContext(conf=sparkConf)

# In this example, the walk will last 30 steps
# As we chose a `Line` mesh, its size will be
# automatically calculated, i.e., 2 * size + 1 sites
size = steps = 30

# Choosing a coin and a mesh for the walk
coin = Hadamard()
mesh = Line([size])

mesh_size = mesh.size[0]

# Center of the mesh
# Notice that we set a list with only one element
# as we are simulating a DTQW with one particle
positions = [mesh.center()]

# Options of initial states
# Notice that we set a list with only one element
# as we are simulating a DTQW with one particle
# |c>|p> --> (|0>|p> - i|1>|p>) / sqrt(2)
amplitudes = [[(1.0 + 0.0j) / math.sqrt(2),
               (0.0 - 1.0j) / math.sqrt(2)]]

# Building the initial state
initial_state = State.create(coin, mesh, positions, amplitudes)

# Instantiating the walk
dtqw = DiscreteTimeQuantumWalk(initial_state)

# Performing the walk
final_state = dtqw.walk(steps)

# Measuring the state of the system and plotting its probability distribution
gauge = PositionGauge()

joint = gauge.measure(final_state)

plot.line(mesh.axis(), joint.ndarray(), 'joint_1d1p',
          labels=['Position', 'Probability'])

# Destroying the RDD and stopping the SparkContext
joint.destroy()
final_state.destroy()
dtqw.destroy_operators()
initial_state.destroy()

sparkContext.stop()

```

For more detailed examples (e.g., of how to use profiling and mesh percolations), the user may refer to the files in the `examples` directory.

#### Coins and Meshes

By default, the following coins and meshes have already been implemented:

- For one-dimensional walks:
  - Coin:
    - `Hadamard`
  - Mesh:
    - `Line`: a mesh based on the number of steps of the walk. To avoid the particles walking besides the boundaries of the mesh, its size is the double of the number of steps plus a center site where the particles **must** be located initially for a flawless simulation;
    - `Segment`: a line-based mesh (although there is no relation between the size of the mesh and the number of steps of the walk) with reflective sites on each border. The particles can start their walk at any site of the mesh;
    - `Cycle`: a line-based mesh (although there is no relation between the size of the mesh and the number of steps of the walk) with cyclic sites on each border. The particles can start their walk at any site of the mesh
- For two-dimensional walks:
  - Coin:
    - `Hadamard`
    - `Grover`
    - `Fourier`
  - Mesh (each one with diagonal and natural variants):
    - `Lattice`: a mesh based on the number of steps of the walk. To avoid the particles walking besides the boundaries of the mesh, its size is the double of the number of steps plus a center site where the particles **must** be located initially for a flawless simulation. It's the `Line`'s two-dimension counterpart;
    - `Box`: a lattice-based mesh (although there is no relation between the size of the mesh and the number of steps of the walk) with reflective sites on each coordinates' border. The particles can start their walk at any site of the mesh. It's the `Segment`'s two-dimension counterpart;
    - `Torus`: a lattice-based mesh (although there is no relation between the size of the mesh and the number of steps of the walk) with cyclic sites on each coordinates' border. The particles can start their walk at any site of the mesh. It's the `Cycle`'s two-dimension counterpart

#### Mesh Percolations

The simulator lets the user simulate DTQWs with some mesh percolations. The already implemented variations are "random" and "permanent". For the former, the user must instantiate its corresponding class (`RandomBrokenLinks`) passing in the probability value that will be used to generate the broken edges of the mesh in a random fashion:

```python
broken_links = RandomBrokenLinks(0.05)
```

and assign it to the chosen mesh, as follows:

```python
mesh = Line(steps, broken_links)
```

The second one is represented by the `PermanentBrokenLinks` class. Its usage differs from the first variation only in the parameter it receives, which is a list with the number of each edge that is broken:

```python
broken_links = PermanentBrokenLinks([5, 55])
```

In order to correctly inform the number of the edges, the user must know how the simulator numbers them: starting with the one-dimensional meshes, the edges are incrementally numbered following a left-to-right direction, starting with the leftmost edge. The last edge has the same number of the first one, as if it was a cycled mesh, to consider the border extrapolation:

```
— o — o — o — o — o —
0   1   2   3   4   0
          x
```

For two-dimensional meshes, the previous principle is also used, although some adaptations must be performed. When considering the diagonal mesh, as the particle moves only diagonally, the number of sites that can be occupied by the particle is inferior than the sites of the mathematical mesh. Also, notice that the number of edges traversed by the particle equals the number of positions of the grid:

```
\     / \     / \     /
 0   1   2   3   4   0
  \ /     \ /     \ /
   o   ∙   o   ∙   o
  / \     / \     / \
20  21  22   23  24  20
/     \ /     \ /     \
   ∙   o   ∙   o   ∙
\     / \     / \     /
15  16  17   18  19  15
  \ /     \ /     \ /
   o   ∙   o   ∙   o     y
  / \     / \     / \
10  11  12   13  14  10
/     \ /     \ /     \
   ∙   o   ∙   o   ∙
\     / \     / \     /
 5   6   7   8   9   5
  \ /     \ /     \ /
   o   ∙   o   ∙   o
  / \     / \     / \
 0   1   2   3   4   0
/     \ /     \ /     \
           x
```

When a natural mesh is considered, all the possible positions that the particle can be located at coincide with the mathematical grid, resulting in a higher number of positions in relation to the previous case, and being the double of the number of edges traversed by the particle. The edge numbering is done for both directions separately, starting with the horizontal (_x_ coordinate) and then, with the vertical (_y_ coordinate):

```
       |        |        |        |        |
      25       30       35       40       45
       |        |        |        |        |
— 20 — o — 21 — o — 22 — o — 23 — o — 24 — o — 20 —
       |        |        |        |        |
      29       34       39       44       49
       |        |        |        |        |
— 25 — o — 26 — o — 27 — o — 28 — o — 29 — o — 15 —
       |        |        |        |        |
      28       33       38       43       48
       |        |        |        |        |
— 10 — o — 11 — o — 12 — o — 13 — o — 14 — o — 10 — y
       |        |        |        |        |
      27       32       37       42       47
       |        |        |        |        |
— 05 — o — 06 — o — 07 — o — 08 — o — 09 — o — 05 —
       |        |        |        |        |
      26       31       36       41       46
       |        |        |        |        |
— 00 — o — 01 — o — 02 — o — 03 — o — 04 — o — 0 —
       |        |        |        |        |
      25       30       35       40       45
       |        |        |        |        |
                         x
```

#### Custom Elements

TODO: explain how the user can implement custom coins, meshes and custom mesh percolations.

## Configuration Parameters

Below, there is a list of the current configuration parameters that the user can define accodingly to his/her needs. The values must be set through `SparkConf` key-value pairs:

| Property Name                                             |                      Default                      | Meaning                                                                                                                                                          |
| :-------------------------------------------------------- | :-----------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sparkquantum.cluster.maxPartitionSize                     |                       64MB                        | Maximum partition size of each RDD produced by the application. Only integers are accepted.                                                                      |
| sparkquantum.cluster.numPartitionsSafetyFactor            |                        1.3                        | A safety factor when calculating the best possible number of partitions of each RDD produced by the application.                                                 |
| sparkquantum.cluster.totalCores                           |                         1                         | Total number of cores used by the application. Necessary to calculate the best possible number of partitions of each RDD produced by the application             |
| sparkquantum.cluster.useSparkDefaultNumPartitions         |                       False                       | Whether to use the default number of partitions defined by Spark when partitioning the RDD produced by the application.                                          |
| sparkquantum.dtqw.interactionOperator.checkpoint          |                       False                       | Whether to checkpoint the interaction operator. Considered only on interacting multiparticle walks.                                                              |
| sparkquantum.dtqw.mesh.brokenLinks.generationMode         |  `constants.BrokenLinksGenerationModeBroadcast`   | The broken links generation mode. For now, can be as a broadcast variable or a RDD, both ways containing the edges numbers that are broken.                      |
| sparkquantum.dtqw.profiler.logExecutors                   |                       False                       | Whether to log executors' data if a profiler was provided.                                                                                                       |
| sparkquantum.dtqw.state.dumpingFormat                     |        `constants.StateDumpingFormatIndex`        | Whether the system state has each of its elements dumped as vector indexes followed by their values or as mesh/cartesian coordinates followed by their values.   |
| sparkquantum.dtqw.state.representationFormat              | `constants.StateRepresentationFormatCoinPosition` | Whether the system state is represented by a kronecker product between the coin and position spaces or between the position and coin spaces.                     |
| sparkquantum.dtqw.walk.checkpointingFrequency             |                        -1                         | The frequency to checkpoint the states. A state will be checkpointed at every _n_ steps. When -1, it never checkpoints the state.                                |
| sparkquantum.dtqw.walk.checkUnitary                       |                       False                       | Whether to check if each state is unitary.                                                                                                                       |
| sparkquantum.dtqw.walk.dumpingFrequency                   |                        -1                         | The frequency of dumping the states to disk. A state will be dumped at every _n_ steps. When -1, it never dumps the state.                                       |
| sparkquantum.dtqw.walk.dumpingPath                        |                       "./"                        | The directory to save the dumps.                                                                                                                                 |
| sparkquantum.dtqw.walk.dumpStatesProbabilityDistributions |                       False                       | Whether to dump to disk the probability distribution of each state.                                                                                              |
| sparkquantum.dtqw.walkOperator.checkpoint                 |                       False                       | Whether to checkpoint the walk operator(s).                                                                                                                      |
| sparkquantum.dtqw.walkOperator.kroneckerMode              |        `constants.KroneckerModeBroadcast`         | The kronecker product mode to build the walk operator(s). For now, can be as a broadcast variable or a RDD.                                                      |
| sparkquantum.dtqw.walkOperator.tempPath                   |                       "./"                        | The temporary directory to save the walk operators' dump. Considered only when the kronecker mode is `constants.KroneckerModeDump`.                              |
| sparkquantum.dumpingCompressionCodec                      |                       None                        | Compression codec class used by Spark when dumping each RDD's data disk.                                                                                         |
| sparkquantum.dumpingGlue                                  |                        " "                        | A string to connect the coordinates of each RDD's element when dumping its data.                                                                                 |
| sparkquantum.logging.enabled                              |                       False                       | Whether the application must use [Python's logging facility](https://docs.python.org/3/library/logging.html).                                                    |
| sparkquantum.logging.filename                             |                    './log.txt'                    | The filename (with relative or absolute path) where all log data will be written by Python's logging facility.                                                   |
| sparkquantum.logging.format                               | '%(levelname)s:%(name)s:%(asctime)s:%(message)s'  | Python's logging.Formatter acceptable format which each log record will be at.                                                                                   |
| sparkquantum.logging.level                                |                 `logging.WARNING`                 | Python's logging facility acceptable severity level.                                                                                                             |
| sparkquantum.math.dumpingMode                             |         `constants.DumpingModePartFiles`          | Whether the mathematical entity's (Matrix, Vector, Operator, State, etc.) RDD has its data dumped to disk in a single file or in multiple part-\* files in disk. |
| sparkquantum.math.roundPrecision                          |                        10                         | Decimal precision when rounding numbers.                                                                                                                         |
| sparkquantum.profiling.enabled                            |                       False                       | Whether the application must profile all created RDD to get their metrics.                                                                                       |
| sparkquantum.profiling.baseUrl                            |          'http://localhost:4040/api/v1/'          | The [Spark Rest API](http://spark.apache.org/docs/latest/monitoring.html#rest-api) base URL that the application's profiler must use to get some metrics.        |
