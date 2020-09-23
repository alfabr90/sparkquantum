# sparkquantum

The idea of this project (perhaps it's still needed a better name for it) is to provide to the community a quantum algorithms simulator written in [Python](https://www.python.org/) using [Apache Spark](https://spark.apache.org/).

As they evolve, some quantum algorithms simulations acquire characteristics of a Big Data application due to the exponential growth suffered by their data structures. Thus, employing a framework like Apache Spark is a good approach, making it possible to execute larger simulations in high-performance computing (HPC) environments than when using single-processors, general purpose computers, allowing such data to be generated and processed at a reduced time in a parallel/distributed way.

## Requirements

To run the simulations, the [numpy](http://www.numpy.org/) package must be installed in the user's environment. Also, to plot the probability distributions, the user must install the [matplotlib](https://matplotlib.org/) package.

The [pyspark](https://pypi.org/project/pyspark/) package is also required, but instead of installing it in the user's environment (which is an alternative), he/she may configure and launch Spark in such way that Python can access the package provided in Spark's directory.

## How to Use It

In contrast to other simulators, this one expects a programmatic way to input the initial conditions of the simulations, but its use is still easy and simple, requiring few steps to start using it.

For now, the user can only simulate discrete time quantum walks (DTQW), even though _sparkquantum_ allows the simulation of any unitary sequence of operations. Simulations of Grover's search and Shor's integer factorizarion algorithms are planned to be implemented soon.

To check all source code documentation (Python docstrings), visit this project's page in [Read the Docs](https://sparkquantum.readthedocs.io/en/latest/).

### Discrete Time Quantum Walk

As DTQW is characterized as an iterative algorithm, this simulator considers each step as a matrix-vector multiplication - the matrix represents the unitary evolution operator and the vector represents the current state of the system.

The `dtqw` module of the simulator allows the user to simulate DTQW composed by _n_ particles over any kind of mesh (see the list of available meshes already implemented), with or without mesh percolations (broken links), and, in the end, measure the final quantum state so that a probability distribution of the particles' possible positions can be obtained and plotted.

#### A Simple Example

In order to simulate, for instance, a particle walking over a line (one-dimensional grid), the user must, first of all, instantiate this mesh's corresponding class informing its size (shape):

```python
mesh = Line((2 * steps + 1, ))
```

Particularly for the line (one-dimensional) grid, its size comprehends a central site where the particle **must**, initially, be located at, and the number of steps to the left and to the right of that central site. This avoids the particle walking besides the boundaries of the mesh.

Notice that, due to the mesh being a one-dimensional grid, its shape must be defined by a one-element tuple. For other dimensions, e.g. two-dimensional, the correspondent shape must be defined by a two-elements tuple and so on.

Next, the user must instantiate a `DiscreteTimeQuantumWalk` object with the previously chosen mesh:

```python
dtqw = DiscreteTimeQuantumWalk(mesh)
```

From now on, particles can be added to the quantum walk, but first, a coin must be instantiated with the mesh's correspondent dimension number:

```python
coin = Hadamard(mesh.ndim)
```

When instantiating a particle, the user can assign an identifier to it:

```python
particle = Particle(coin, identifier='Electron')
```

In this example, suppose the user needs to simulate the following initial quantum state, in Dirac's notation:

`|c>|p> --> (|0> - i|1>)|p> / sqrt(2) = (|0>|p> - i|1>|p>) / sqrt(2)`,

where `c` is the coin state and `p` is the position state. In order to do so, the user need to add the particle to the quantum walk, informing its coin state and position as follows:

```python
dtqw.add_particle(particle, (1 / math.sqrt(2), 1j / math.sqrt(2)), mesh.center())
```

To perform the walk, the user must call the `DiscreteTimeQuantumWalk.walk` method, informing the desired number of steps:

```python
state = dtqw.walk(steps)
```

Finally, the user can measure the final quantum state, obtaining the probability distribution of the possible particle's positions and plot it:

```python
joint = Position().measure(state)

labels = ["{}'s position x".format(particle.identifier), 'Probability']
joint.plot(path + 'joint_1d1p', labels=labels)

```

Below, the complete Python script of the previous example with some Spark related commands:

```python
import math

from pyspark import SparkContext, SparkConf

from sparkquantum import constants, plot, util
from sparkquantum.dtqw.coin.hadamard import Hadamard
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.dtqw.mesh.grid.onedim.line import Line
from sparkquantum.dtqw.observable.position import Position
from sparkquantum.dtqw.particle import Particle

# Choosing a directory to store plots
path = './output/dtqw/'
util.create_dir(path)

# Initiallizing the SparkContext with some options
conf = SparkConf().set('sparkquantum.cluster.totalCores', 4)
sc = SparkContext(conf=conf)

# In this example, the walk will last 100 steps.
steps = 100

# As we chose a `Line` mesh, its size must be
# 2 * steps + 1 sites
size = 2 * steps + 1

# Choosing a mesh and instantiating the walk with the chosen mesh
mesh = Line((size, ))
dtqw = DiscreteTimeQuantumWalk(mesh)

# To add particles to the walk, a coin must be instantiated with
# the correspondent dimension of the chosen mesh
coin = Hadamard(mesh.ndim)

# Instantiating a particle and giving it a name
particle = Particle(coin, identifier='Electron')

# Options of initial coin states
# |i> --> (|0> - i|1>) / sqrt(2)
cstate = (1 / math.sqrt(2), 1j / math.sqrt(2))

# |i> --> |0>
# cstate = (1, 0)

# |i> --> |1>
# cstate = (0, 1)

# Adding the particle to the walk with its coin state and
# position corresponding to the center site of the mesh
dtqw.add_particle(particle, cstate, mesh.center())

# Performing the walk
state = dtqw.walk(steps)

# Measuring the state of the system and plotting its distribution
joint = Position().measure(state)

labels = ["{}'s position x".format(particle.identifier), 'Probability']
joint.plot(path + 'joint_1d1p', labels=labels)

# Destroying the RDD and stopping the SparkContext
state.destroy()
dtqw.destroy()
joint.destroy()

# Stopping the SparkContext
sc.stop()

```

For more detailed examples, the user may refer to the files in the `examples` directory.

#### Coins and Meshes

By default, the following coins have been implemented:

- Coin:
  - `Hadamard`
  - `Grover` - must be two-dimensional or higher;
  - `Fourier` - must be two-dimensional or higher;

and meshes:

- For one-dimensional walks:
  - `Line`: a grid based on the number of steps of the walk. To avoid the particles walking besides the boundaries of the grid, its size must be the double of the number of steps plus a center site where the particles **must** be located initially for a flawless simulation;
  - `Segment`: a line-based grid (although there is no relation between the size of the grid and the number of steps of the walk) with reflective sites on each border. The particles can start their walk at any site of the grid;
  - `Cycle`: a line-based grid (although there is no relation between the size of the grid and the number of steps of the walk) with cyclic sites on each border. The particles can start their walk at any site of the grid;
- For two-dimensional walks (each one with diagonal and natural variants):
  - `Lattice`: a grid based on the number of steps of the walk. To avoid the particles walking besides the boundaries of the grid, its size must be the double of the number of steps plus a center site where the particles **must** be located initially for a flawless simulation. It's the `Line`'s two-dimension counterpart;
  - `Box`: a lattice-based grid (although there is no relation between the size of the grid and the number of steps of the walk) with reflective sites on each coordinates' border. The particles can start their walk at any site of the grid. It's the `Segment`'s two-dimension counterpart;
  - `Torus`: a lattice-based grid (although there is no relation between the size of the grid and the number of steps of the walk) with cyclic sites on each coordinates' border. The particles can start their walk at any site of the grid. It's the `Cycle`'s two-dimension counterpart.

#### Mesh Percolations

_sparkquantum_ lets the user simulate DTQWs with some mesh percolations. The already implemented variations are "random" and "permanent". For the former, the user must instantiate its corresponding class (`Random`) passing in the probability value that will be used to generate the broken edges of the mesh in a random fashion:

```python
percolation = Random(0.05)
```

and assign it to the chosen mesh, as follows:

```python
mesh = Line(steps, percolation=percolation)
```

The second one is represented by the `Permanent` class. Its usage differs from the first variation only in the parameter it receives, which is a collection with the number of each edge that is broken:

```python
percolation = Permanent([5, 55])
```

In order to correctly inform the number of the edges, the user must know how the simulator numbers them: starting with the one-dimensional grids, the edges are incrementally numbered following a left-to-right direction, starting with the leftmost edge. The last edge has the same number of the first one, as if it was a cycled mesh, to consider the border extrapolation:

```
— o — o — o — o — o —
0   1   2   3   4   0
          x
```

For two-dimensional grids, the previous principle is also used, although some adaptations must be performed. When considering the diagonal mesh, as the particle moves only diagonally, the number of sites that can be occupied by the particle is inferior than the sites of the mathematical grid. Also, notice that the number of edges traversed by the particle equals the number of positions of the grid:

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

#### Enabling Logging and Profiling

In some cases, the user may want to know how a long lasting simulation is being executed. To accomplish that, _sparkquantum_ optionally exposes a log file that keeps being filled with some messages and/or data related to each simulation step. This log file is generated and filled using [Python's logging facility](https://docs.python.org/3/library/logging.html).

To enable logging, set the `sparkquantum.logging.enabled` configuration parameter with `'True'` as follows:

```python
conf = SparkConf() \
    # other configurations
    .set('sparkquantum.logging.enabled', 'True')
    # other configurations
```

_sparkquantum_ allows customizing some Python's logging options the same way as above. The possible configurations are described in the table in the end of this documentation.

Besides logging, the simulator can also gathers data about the resource usage during the simulations and exports them in CSV files. Similarly as the previous feature, the user may enable profiling through setting one configuration parameter, namely `sparkquantum.profiling.enabled`:

```python
conf = SparkConf() \
    # other configurations
    .set('sparkquantum.profiling.enabled', 'True')
    # other configurations
```

To export all gathered profiling data, the user just need to call, at the end of the simulation, the method `export` of the profiler that is automatically attached to the `DiscreteTimeQuantumWalk` and `Position` (`Observable`) classes:

```python
# Exporting the profiling data
dtqw.profiler.export(path)
position.profiler.export(path)
```

where `dtqw` and `position` are, respectively, the discrete time quantum walk and position observable objects.

#### Custom Elements

TODO: explain how the user can implement custom coins, meshes and custom mesh percolations and interactions between particles.

## Configuration Parameters

Below, there is a list of the current configuration parameters that the user can define accodingly to his/her needs. The values must be set through `SparkConf` key-value pairs:

| Property Name                                             |                      Default                      | Meaning                                                                                                                                                          |
| :-------------------------------------------------------- | :-----------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sparkquantum.cluster.maxPartitionSize                     |                       64MB                        | Maximum partition size of each RDD produced by the application. Only integers are accepted.                                                                      |
| sparkquantum.cluster.numPartitionsSafetyFactor            |                        1.3                        | A safety factor when calculating the best possible number of partitions of each RDD produced by the application.                                                 |
| sparkquantum.cluster.totalCores                           |                         1                         | Total number of cores used by the application. Necessary to calculate the best possible number of partitions of each RDD produced by the application             |
| sparkquantum.cluster.useSparkDefaultNumPartitions         |                       False                       | Whether to use the default number of partitions defined by Spark when partitioning the RDD produced by the application.                                          |
| sparkquantum.dtqw.interactionOperator.checkpoint          |                       False                       | Whether to checkpoint the interaction operator. Considered only on interacting multiparticle walks.                                                              |
| sparkquantum.dtqw.mesh.percolation.generationMode         |  `constants.PercolationsGenerationModeBroadcast`  | The broken links generation mode. For now, can be as a broadcast variable or a RDD, both ways containing the edges numbers that are broken.                      |
| sparkquantum.dtqw.profiler.logExecutors                   |                       False                       | Whether to log executors' data if a profiler was provided.                                                                                                       |
| sparkquantum.dtqw.state.dumpingFormat                     |        `constants.StateDumpingFormatIndex`        | Whether the system state has each of its elements dumped as vector indexes followed by their values or as mesh/cartesian coordinates followed by their values.   |
| sparkquantum.dtqw.stateRepresentationFormat               | `constants.StateRepresentationFormatCoinPosition` | Whether the system state is represented by a kronecker product between the coin and position spaces or between the position and coin spaces.                     |
| sparkquantum.dtqw.walk.checkpointingFrequency             |                        -1                         | The frequency to checkpoint the states. A state will be checkpointed at every _n_ steps. When -1, it never checkpoints the state.                                |
| sparkquantum.dtqw.walk.checkUnitary                       |                       False                       | Whether to check if each state is unitary.                                                                                                                       |
| sparkquantum.dtqw.walk.dumpingFrequency                   |                        -1                         | The frequency of dumping the states to disk. A state will be dumped at every _n_ steps. When -1, it never dumps the state.                                       |
| sparkquantum.dumpingPath                                  |                       "./"                        | The directory to save the dumps.                                                                                                                                 |
| sparkquantum.dtqw.walk.dumpStatesProbabilityDistributions |                       False                       | Whether to dump to disk the probability distribution of each state.                                                                                              |
| sparkquantum.dtqw.evolutionOperator.checkpoint            |                       False                       | Whether to checkpoint the walk operator(s).                                                                                                                      |
| sparkquantum.dtqw.evolutionOperator.kroneckerMode         |        `constants.KroneckerModeBroadcast`         | The kronecker product mode to build the walk operator(s). For now, can be as a broadcast variable or a RDD.                                                      |
| sparkquantum.dtqw.evolutionOperator.tempPath              |                       "./"                        | The temporary directory to save the walk operators' dump. Considered only when the kronecker mode is `constants.KroneckerModeDump`.                              |
| sparkquantum.dumpingCompressionCodec                      |                       None                        | Compression codec class used by Spark when dumping each RDD's data disk.                                                                                         |
| sparkquantum.dumpingGlue                                  |                        " "                        | A string to connect the coordinates of each RDD's element when dumping its data.                                                                                 |
| sparkquantum.logging.enabled                              |                       False                       | Whether the application must generate and fill a log file.                                                                                                       |
| sparkquantum.logging.filename                             |                    './log.txt'                    | The filename (with relative or absolute path) where all log data will be written by Python's logging facility.                                                   |
| sparkquantum.logging.format                               | '%(levelname)s:%(name)s:%(asctime)s:%(message)s'  | Python's logging.Formatter acceptable format which each log record will be at.                                                                                   |
| sparkquantum.logging.level                                |                 `logging.WARNING`                 | Python's logging facility acceptable severity level.                                                                                                             |
| sparkquantum.math.dumpingMode                             |         `constants.DumpingModePartFiles`          | Whether the mathematical entity's (Matrix, Vector, Operator, State, etc.) RDD has its data dumped to disk in a single file or in multiple part-\* files in disk. |
| sparkquantum.math.roundPrecision                          |                        10                         | Decimal precision when rounding numbers.                                                                                                                         |
| sparkquantum.profiling.enabled                            |                       False                       | Whether the application must profile all created RDD to get their metrics.                                                                                       |
| sparkquantum.profiling.baseUrl                            |          'http://localhost:4040/api/v1/'          | The [Spark Rest API](http://spark.apache.org/docs/latest/monitoring.html#rest-api) base URL that the application's profiler must use to get some metrics.        |
