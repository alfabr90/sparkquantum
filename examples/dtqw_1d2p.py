import math
import cmath

from pyspark import SparkContext, SparkConf

from sparkquantum import constants, plot, util
from sparkquantum.dtqw.coin.hadamard import Hadamard
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.dtqw.interaction.collision.phase import PhaseChange
from sparkquantum.dtqw.mesh.grid.onedim.line import Line
from sparkquantum.dtqw.observable.position import Position
from sparkquantum.dtqw.particle import Particle

base_path = './output'
cores = 4

particles = 2
entangled = True

# In this example, the walk will last 30 steps.
# As we chose a `Line` mesh, its size must be
# 2 * steps + 1 sites
steps = 30
size = 2 * steps + 1

# The particles will change their phase when colliding
phase = 1.0 * cmath.pi

# Choosing a directory to store plots and logs
path = "{}/{}_{}_{}_{}_{}_{}_{}/".format(
    base_path, 'hadamard', 'line', size, steps, particles, phase,
    'entangled' if entangled else 'not-entangled'
)
util.create_dir(path)

# Initiallizing the SparkContext with some options
conf = SparkConf().set('sparkquantum.cluster.totalCores', cores)
sc = SparkContext(conf=conf)
sc.setLogLevel('ERROR')

# Choosing a mesh and instantiating the interacting walk with it
mesh = Line((size, ))
dtqw = DiscreteTimeQuantumWalk(
    mesh,
    interaction=PhaseChange(phase),
    repr_format=constants.StateRepresentationFormatCoinPosition)

# To add particles to the walk, a coin must be instantiated with
# the correspondent dimension of the chosen mesh
coin = Hadamard(mesh.ndim)

# Instantiating the particle and giving them an identifier/name
particle1 = Particle(coin, identifier='Fermion')
particle2 = Particle(coin, identifier='Boson')

# Options of initial coin states for the particle
# |i> --> (|0> - i|1>) / sqrt(2)
cstate = (1 / math.sqrt(2), 1j / math.sqrt(2))

# |i> --> |0>
# cstate = (1, 0)

# |i> --> |1>
# cstate = (0, 1)

position = mesh.center()

if not entangled:
    # Adding the particles to the walk, with their coin state and
    # position corresponding to the center site of the mesh
    dtqw.add_particle(particle1, cstate, position)
    dtqw.add_particle(particle2, cstate, position)
else:
    # The coin space for one-dimensional grids has size of 2
    cspace = 2

    # The position space has size corresponding to the number of sites
    pspace = mesh.sites

    if dtqw.repr_format == constants.StateRepresentationFormatCoinPosition:
        # |i1>|x1>|i2>|x2> --> (|1>|x1>|0>|x2> - |0>|x1>|1>|x2>) / sqrt(2)
        state = [[(1 * pspace + position) * cspace * pspace + 0 * pspace + position, 1, 1.0 / math.sqrt(2)],
                 [(0 * pspace + position) * cspace * pspace + 1 * pspace + position, 1, -1.0 / math.sqrt(2)]]

        # |i1>|x1>|i2>|x2> --> (|1>|x1>|0>|x2> + |0>|x1>|1>|x2>) / sqrt(2)
        # state = [[(1 * pspace + position) * cspace * pspace + 0 * pspace + position, 1, 1.0 / math.sqrt(2)],
        #          [(0 * pspace + position) * cspace * pspace + 1 * pspace + position, 1, 1.0 / math.sqrt(2)]]
    elif dtqw.repr_format == constants.StateRepresentationFormatPositionCoin:
        # |x1>|i1>|x2>|i2> --> (|x1>|1>|x2>|0> - |x1>|0>|x2>|1>) / sqrt(2)
        state = [[(position * cspace + 1) * pspace * cspace + position * cspace + 0, 1, 1.0 / math.sqrt(2)],
                 [(position * cspace + 0) * pspace * cspace + position * cspace + 1, 1, -1.0 / math.sqrt(2)]]

        # |x1>|i1>|x2>|i2> --> (|x1>|1>|x2>|0> + |x1>|0>|x2>|1>) / sqrt(2)
        # state = [[(position * cspace + 1) * pspace * cspace + position * cspace + 0, 1, 1.0 / math.sqrt(2)],
        #          [(position * cspace + 0) * pspace * cspace + position * cspace + 1, 1, 1.0 / math.sqrt(2)]]

    # Adding the entangled particles to the walk, informing their system state
    dtqw.add_entanglement((particle1, particle2), state)

# Performing the walk
state = dtqw.walk(steps)

# Measuring the state of the system and plotting its distribution
position = Position()

joint, collision, marginal = position.measure(state)

labels = ["{}'s position x".format(particle1.identifier),
          "{}'s position x".format(particle2.identifier),
          'Probability']
joint.plot(path + 'joint_1d2p', labels=labels, dpi=300)

labels = ["Particles' position x", 'Probability']
collision.plot(path + 'collision_1d2p', labels=labels, dpi=300)

for p in range(particles):
    labels = ["{}'s position x".format(dtqw.particles[p].identifier),
              'Probability']
    marginal[p].plot(path + 'marginal_1d2p_particle{}'.format(p + 1),
                     labels=labels, dpi=300)

# Destroying the RDD and stopping the SparkContext
state.destroy()
dtqw.destroy()
joint.destroy()
collision.destroy()
for m in marginal:
    m.destroy()
sc.stop()
