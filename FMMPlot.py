from time import time
from functools import wraps, partial
import numpy as np
from copy import deepcopy
import FMMFunc as fmm
import matplotlib.pyplot as plt

#  Time complexity testing + code to plot


def time_function(func):
    """
    Decorator to measure the runtime of a function.

    Args:
        func (callable): The function to be timed.

    Returns:
        callable: A decorated function that measures its runtime.
    """
    @wraps(func)
    def timed_function(*args, **kwargs):
        """Wrapper function to measure the runtime of the decorated function."""
        start = time()
        res = func(*args, **kwargs)
        runtime = (time() - start)
        return runtime
    return timed_function

# Decorate functions to measure their runtime
fmmtimedf = time_function(partial(fmm.potential, tree_thresh=10))
directtimed = time_function(fmm.potentialDS)

# Generate random particles for testing
particles = [fmm.Particle(*p, 1) for p in np.random.rand(100,2)]

# Define number of particles for testing
num_particles = (100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000)

# pinit function used for a random distribution of particles
pinit = lambda n: [fmm.Particle(*p, 1) for p in np.random.rand(n,2)]

'''
# Modify pinit to generate particles with a uniform distribution
def pinit_uniform(n):
    """
    Generate particles with a uniform distribution in a square grid.

    Args:
        n (int): Number of particles to generate.

    Returns:
        list: List of Particle objects with uniform distribution.
    """
    side_length = int(np.ceil(np.sqrt(n)))  # Adjusting side length to fit all particles
    particles = []
    step = 1.0 / side_length
    for i in range(side_length):
        for j in range(side_length):
            x = (i + 0.5) * step  # Adding 0.5 to center particles in each grid cell
            y = (j + 0.5) * step
            if len(particles) < n:
                particles.append(Particle(x, y, 1))  # Assuming charge of 1 for all particles
    return particles

# Use the modified pinit function to generate particles
pinit = pinit_uniform
'''

# Measure runtime for different numbers of particles
t_fmmf = list(map(fmmtimedf, map(pinit, num_particles)))
t_direct = list(map(directtimed, map(pinit, num_particles)))

# Plot the results
num_particles = np.array(num_particles)
sizen = num_particles / num_particles[0]
plt.semilogy(num_particles, t_direct[0] * (sizen) ** 2)
plt.semilogy(num_particles, t_fmmf[0] * sizen * np.log(np.e * sizen))
plt.semilogy(num_particles, t_fmmf[0] * sizen)
plt.semilogy(num_particles, t_fmmf, 'o--')
plt.semilogy(num_particles, t_direct, 'o--')
plt.legend([r'O(n$^2$)', 'O(nlog(n))', 'O(n)', 'FMM', 'Direct'])
plt.xlabel('Number of Particles')
plt.ylabel('Runtime (s)')
plt.tight_layout()
plt.savefig('FMM Random.png', dpi=300)
plt.show()
