import numpy as np
import matplotlib.pyplot as plt
import timeit

# Electrostatic Potential BRUTE

def calculate_potential_2d(positions, charges):
    num_particles = len(positions)
    potential = np.zeros(num_particles)

    for i in range(num_particles):
        for j in range(num_particles):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                potential[i] += charges[j] / distance

    return potential

def time_potential_calculation_2d(num_particles):
    # Generate random positions and charges for particles
    # positions = np.random.random((num_particles, 2))  # Random positions in 2D space
    # charges = np.random.random(num_particles)         # Random charges

    # Generate ordered uniform positions and charges for particles
    side_length = int(np.sqrt(num_particles))  # Assuming a square grid
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, side_length), np.linspace(0, 1, side_length))
    x_positions = x_grid.flatten()
    y_positions = y_grid.flatten()
    positions = np.column_stack((x_positions, y_positions))  # Combine x and y positions
    charges = np.random.random(num_particles)  # Random charges

    # Measure the time taken for potential calculation
    start_time = timeit.default_timer()  # Start timer
    calculate_potential_2d(positions, charges)
    end_time = timeit.default_timer()    # End timer

    runtime = end_time - start_time      # Calculate runtime
    return runtime

# Barnes-Hut Method Electrostatic Potential

class Node:
    """
    Represents a node within the quadtree used in the Barnes-Hut algorithm.

    Attributes:
        pos (np.array): Position vector representing the center of mass.
        pot (float): Electrostatic potential at the node.
        q (float): Charge of the node.
        child (list): Child nodes if any, forming the quadtree structure.
        s (float): Side length of the quadrant (depth=0, s=1).
        relpos (np.array): Relative position vector within the quadrant.
    """

    def __init__(self, x, y, q):
        """
        Initializes a node without children.

        Args:
            x (float): x-coordinate of the position.
            y (float): y-coordinate of the position.
            q (float): Charge of the node.
        """
        self.q = q
        self.pos = np.array([x, y], dtype=float)
        self.pot = 0
        self.child = None

    def next_quad(self):
        """
        Moves the node to the next quadrant and returns the quadrant number.
        """
        self.s = 0.5 * self.s
        return self.divide_quad(1) + 2 * self.divide_quad(0)

    def divide_quad(self, i):
        """
        Divides the node into the next level quadrant and recomputes relative position.

        Args:
            i (int): Index of the dimension to divide.

        Returns:
            int: 0 if the node is in the first half, 1 if in the second half.
        """
        self.relpos[i] *= 2.0
        if self.relpos[i] < 1.0:
            return 0
        else:
            self.relpos[i] -= 1.0
            return 1

    def reset_quad(self):
        """
        Resets the node to the zeroth depth quadrant (full space).
        """
        self.s = 1.0
        self.relpos = self.pos.copy()

    def dist(self, other):
        """
        Calculates the distance between this node and another node.

        Args:
            other (Node): Another node.

        Returns:
            float: Distance between the two nodes.
        """
        return np.linalg.norm(self.pos - other.pos)

    def potential_at_point(self, point):
        """
        Calculates the electrostatic potential at a given point due to this node.

        Args:
            point (np.array): Position vector of the point.

        Returns:
            float: Electrostatic potential at the point due to this node.
        """
        d = self.dist(Node(*point, 0))  # Create a dummy node at the point
        return self.q / d

def add_charge(charge, node):
    """
    Adds a charge to a node of the quadtree.

    A minimum quadrant size is imposed to limit the recursion depth.

    Args:
        charge (Node): Charge to be added.
        node (Node): Node to which the charge is added.

    Returns:
        Node: Updated node after adding the charge.
    """
    min_quad_size = 1e-5
    while node is not None and node.s > min_quad_size:
        if node.child is None:
            node.child = [None] * 4
            quad = node.next_quad()
            node.child[quad] = charge
            node.pot += charge.pot
            break
        else:
            quad = charge.next_quad()
            node.child[quad] = add_charge(charge, node.child[quad])
            node.pot += charge.pot
            break
    return node

def calculate_potentials(charges, theta):
    """
    Calculates the electrostatic potential for each charge particle using the Barnes-Hut algorithm.

    Args:
        charges (list): List of charge nodes.
        theta (float): Threshold parameter for the Barnes-Hut approximation.

    Returns:
        list: Electrostatic potential for each charge particle.
    """
    potentials = []
    for charge in charges:
        charge.reset_quad()
        potentials.append(potential_at_point_barnes_hut(charge.pos, Root, theta))
    return potentials

def potential_at_point_barnes_hut(point, node, theta):
    """
    Computes the electrostatic potential at a given point using the Barnes-Hut algorithm.

    Args:
        point (np.array): Position vector of the point.
        node (Node): Node from which the potential is calculated.
        theta (float): Threshold parameter for the approximation.

    Returns:
        float: Electrostatic potential at the given point.
    """
    if node is None:
        return 0  # No potential if the node is None

    if node.child is None:
        return node.potential_at_point(point)

    if node.s / node.dist(Node(*point, 0)) < theta:
        return node.potential_at_point(point)

    return sum(potential_at_point_barnes_hut(point, c, theta) for c in node.child if c is not None)

# Main Code

# Parameters
Theta = 0.5 # Barnes-Hut approximation parameter

'''
# Code used to generate a random distribution of N_charges and then time how long it takes for the BH method to run, no plotting
N_charges = 10000 # Number of charges

# Initialize random charges
Charges = [Node(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(N_charges)]

# Build the quadtree
Root = None
for charge in Charges:
    charge.reset_quad()
    Root = add_charge(charge, Root)

# Wrap the calculation into a function
def calculate_potentials_wrapper():
    calculate_potentials(Charges, Theta)

# Time the function, magic function % only works in Google Colab
%timeit calculate_potentials_wrapper()
'''

# Define the Root variable outside the loop
Root = None

# Modify the commented out code to accept a list of N_charges values
def calculate_potentials_wrapper(N_charges):
    # Initialize random charges, if one wanted to plot for a random distribution
    # Charges = [Node(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(N_charges)]

    # Generate an ordered uniform distribution of particles
    side_length = int(np.sqrt(N_charges))  # Assuming a square grid
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, side_length), np.linspace(0, 1, side_length))
    x_positions = x_grid.flatten()
    y_positions = y_grid.flatten()
    Charges = [Node(x, y, np.random.rand()) for x, y in zip(x_positions, y_positions)]

    # Build the quadtree
    global Root # Ensure we're modifying the global Root variable
    Root = None # Reset Root for each new calculation

    for charge in Charges:
        charge.reset_quad()
        Root = add_charge(charge, Root)

    # Calculate potentials
    potentials = calculate_potentials(Charges, Theta)

# Define a range of N_charges values
N_charges_values = [10,20,30,100, 200, 300,400, 500, 1000, 2000, 2500,3500,4500, 5000,6000,7000,7500, 8500,9500,10000]

# Record runtimes for BH for different N_charges values
initial_runtime = None
runtimes = []
for N_charges in N_charges_values:
    runtime = timeit.timeit(lambda: calculate_potentials_wrapper(N_charges), number=1)
    runtimes.append(runtime)
    if initial_runtime is None:
      initial_runtime = runtime

# Record runtimes for different N_charges values using the brute force method
brute_force_runtimes = [time_potential_calculation_2d(N_charges) for N_charges in N_charges_values]

# Generate synthetic runtimes following O(n log n)
def o_n_log_n(n):
    return n * np.log(n)

# Generate synthetic runtimes following O(n^2)
def o_n2(n):
  return n**2

# Generate synthetic runtimes following O(n)
def o_n(n):
  return n

synthetic_runtimes = [o_n_log_n(n) for n in N_charges_values]

synthetic_runtimes2 = [o_n2(n) for n in N_charges_values]

synthetic_runtimes3 = [o_n(n) for n in N_charges_values]

# Calculate the respective scaling factors so that we can normalise the comparison
scaling_factor = initial_runtime / synthetic_runtimes[0]
scaling_factor2 = brute_force_runtimes[0] / synthetic_runtimes2[0]
scaling_factor3 = initial_runtime / synthetic_runtimes3[0]

# Scale the O(n log n) curve by the scaling factor
scaled_synthetic_runtimes = [runtime * scaling_factor for runtime in synthetic_runtimes]

# Scale the O(n^2) curve by the scaling factor
scaled_synthetic_runtimes2 = [runtime * scaling_factor2 for runtime in synthetic_runtimes2]

# Scale the O(n) curve by the scaling factor
scaled_synthetic_runtimes3 = [runtime * scaling_factor3 for runtime in synthetic_runtimes3]

# Plot the graph
plt.plot(N_charges_values, brute_force_runtimes, marker='o', label='Direct')
plt.plot(N_charges_values, runtimes, marker='o', label='BH')
plt.plot(N_charges_values, scaled_synthetic_runtimes, linestyle='--', label='O(n log n)')
plt.plot(N_charges_values, scaled_synthetic_runtimes2, linestyle='--', label='O(n$^2$)')
plt.plot(N_charges_values, scaled_synthetic_runtimes3, linestyle='--', label='O(n)')
# plt.xscale('log')  # Set x-axis to logarithmic scale
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlabel('Number of Charges')
plt.ylabel('Runtime (s)')
plt.legend()
plt.savefig('BH Ordered.png',dpi=300)
plt.show()
