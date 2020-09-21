import math
import cmath

from pyspark import SparkContext, SparkConf

from sparkquantum import constants, plot, util
from sparkquantum.dtqw.coin.hadamard import Hadamard
from sparkquantum.dtqw.dtqw import DiscreteTimeQuantumWalk
from sparkquantum.dtqw.interaction.collision.phase import PhaseChange
from sparkquantum.dtqw.mesh.grid.twodim.diagonal.lattice import Lattice
from sparkquantum.dtqw.mesh.percolation.random import Random
from sparkquantum.dtqw.observable.position import Position
from sparkquantum.dtqw.particle import Particle

base_path = './output'
cores = 4

particles = 2
entangled = True

# In this example, the walk will last 30 steps.
# As we chose a `Lattice` mesh, its size must be
# 2 * steps + 1 sites
steps = 5
size = 2 * steps + 1

# The mesh will have percolations with the following likelihood
probability = 0.3

# The particles will change their phase when colliding
phase = 1.0 * cmath.pi

# Choosing a directory to store plots and logs
path = "{}/{}_{}_{}_{}_{}_{}/".format(
    base_path, 'hadamard', 'diagonal-lattice', size, probability, steps, particles, phase,
    'entangled' if entangled else 'not-entangled'
)
util.create_dir(path)

# Initiallizing the SparkContext with some options
conf = SparkConf().set('sparkquantum.cluster.totalCores', cores)
sc = SparkContext(conf=conf)
sc.setLogLevel('ERROR')

# Choosing a mesh and instantiating the interacting walk with it
mesh = Lattice((size, size), percolation=Random(probability))
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
# |i,j> --> (|0,0> + i|0,1> - i|1,0> + |1,1>) / 2
cstate = (1 / 2, 1j / 2, -1j / 2, 1 / 2)

# |i,j> --> (|0,0> + i|0,1> + i|1,0> - |1,1>) / 2
# cstate = (1 / 2, 1j / 2, 1j / 2, -1 / 2)

# |i,j> --> (|0,0> - |0,1> - |1,0> + |1,1>) / 2
# cstate = (1 / 2, -1 / 2, -1 / 2, 1 / 2)

position = mesh.center()

if not entangled:
    # Adding the particles to the walk, with their coin state and position
    dtqw.add_particle(particle1, cstate, position)
    dtqw.add_particle(particle2, cstate, position)
else:
    # The coin space for two-dimensional grids has size of 4
    cspace = 4

    # The position space has size corresponding to the number of sites
    pspace = mesh.sites

    if dtqw.repr_format == constants.StateRepresentationFormatCoinPosition:
        # |i1,j1>|x1,y1>|i2,j2>|x2,y2> --> (|1,1>|x1,y1>|0,0>|x2,y2> - |0,0>|x1,y1>|1,1>|x2,y2>) / sqrt(2)
        state = [[(3 * pspace + position) * cspace * pspace + (0 * pspace + position), 0, 1.0 / math.sqrt(2)],
                 [(0 * pspace + position) * cspace * pspace + (3 * pspace + position), 0, -1.0 / math.sqrt(2)]]
    elif dtqw.repr_format == constants.StateRepresentationFormatPositionCoin:
        # |x1,y1>|i1,j1>|x2,y2>|i2,j2> --> (|x1,y1>|1,1>|x2,y2>|0,0> - |x1,y1>|0,0>|x2,y2>|1,1>) / sqrt(2)
        state = [[(position * cspace + 3) * pspace * cspace + (position * cspace + 0), 0, 1.0 / math.sqrt(2)],
                 [(position * cspace + 0) * pspace * cspace + (position * cspace + 3), 0, -1.0 / math.sqrt(2)]]

    # Adding the entangled particles to the walk, informing the system state
    dtqw.add_entanglement([particle1, particle2], state)

# Performing the walk
state = dtqw.walk(steps)

# Measuring the state of the system and plotting its distribution
position = Position()

joint, collision, marginal = position.measure(state)

labels = ["Particles' position x", "Particles' position y", 'Probability']
collision.plot(path + 'collision_2d2p', labels=labels, dpi=300)

for p in range(particles):
    labels = ["{}'s position x".format(dtqw.particles[p].identifier),
              "{}'s position y".format(dtqw.particles[p].identifier),
              'Probability']
    marginal[p].plot(path + 'marginal_2d2p_particle{}'.format(p + 1),
                     labels=labels, dpi=300)

# Destroying the RDD and stopping the SparkContext
state.destroy()
dtqw.destroy()
joint.destroy()
collision.destroy()
for p in range(len(marginal)):
    marginal[p].destroy()
sc.stop()
