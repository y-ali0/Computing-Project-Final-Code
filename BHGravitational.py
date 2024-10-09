# Barnes-Hut Method to calculate Gravitational Forces and run an N-Body simulation

import numpy as np

class Node:
    """
    Represents a node within the quadtree used in the Barnes-Hut algorithm.

    Attributes:
        pos (np.array): Position vector representing the center of mass.
        mom (np.array): Momentum vector.
        m (float): Mass of the node.
        child (list): Child nodes if any, forming the quadtree structure.
        s (float): Side length of the quadrant (depth=0, s=1).
        relpos (np.array): Relative position vector within the quadrant.
    """

    def __init__(self, x, y, px, py, m):
        """
        Initializes a node without children.

        Args:
            x (float): x-coordinate of the center of mass.
            y (float): y-coordinate of the center of mass.
            px (float): x-coordinate of momentum.
            py (float): y-coordinate of momentum.
            m (float): Mass of the node.
        """
        self.m = m
        self.pos = np.array([x, y], dtype=float)
        self.mom = np.array([px, py], dtype=float)
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

    def force_ap(self, other):
        """
        Calculates the force applied from this node to another node.

        Args:
            other (Node): Another node.

        Returns:
            np.array: Force vector applied.
        """
        d = self.dist(other)
        return (self.pos - other.pos) * (self.m * other.m / d ** 3)

def add_body(body, node):
    """
    Adds a body to a node of the quadtree.

    A minimum quadrant size is imposed to limit the recursion depth.

    Args:
        body (Node): Body to be added.
        node (Node): Node to which the body is added.

    Returns:
        Node: Updated node after adding the body.
    """
    min_quad_size = 1e-5
    while node is not None and node.s > min_quad_size:
        if node.child is None:
            node.child = [None] * 4
            quad = node.next_quad()
            node.child[quad] = body
            node.m += body.m
            node.pos += body.pos
            break
        else:
            quad = body.next_quad()
            node.child[quad] = add_body(body, node.child[quad])
            node.m += body.m
            node.pos += body.pos
            break
    return node

def force_on(body, node, theta):
    """
    Computes the force acting on a body from a given node and its children
    using the Barnes-Hut algorithm.

    Args:
        body (Node): Body for which force is computed.
        node (Node): Node from which force is calculated.
        theta (float): Threshold parameter for the approximation.

    Returns:
        np.array: Force vector acting on the body.
    """
    if node is None:
        return np.zeros(2)  # No force if the node is None

    if node.child is None:
        return node.force_ap(body)

    if node.s < node.dist(body) * theta:
        return node.force_ap(body)

    return sum(force_on(body, c, theta) for c in node.child if c is not None)

def verlet(bodies, root, theta, G, dt):
    """
    Performs a Verlet integration step for the given bodies.

    Args:
        bodies (list): List of bodies.
        root (Node): Root node of the quadtree.
        theta (float): Threshold parameter for the approximation.
        G (float): Gravitational constant.
        dt (float): Time step for integration.
    """
    for body in bodies:
        force = G * force_on(body, root, theta)
        body.mom += dt * force
        body.pos += dt * body.mom / body.m

def model_step(bodies, theta, G, step):
    """
    Performs a single step of the simulation.

    Args:
        bodies (list): List of bodies.
        theta (float): Threshold parameter for the approximation.
        G (float): Gravitational constant.
        step (float): Time step for integration.
    """
    root = None
    for body in bodies:
        body.reset_quad()
        root = add_body(body, root)
    verlet(bodies, root, theta, G, step)

# Main Code
# Parameters
Theta = 0.5  # Barnes-Hut parameter (0 <= theta <= 1)
G = 1.e-6    # Gravitational constant
dt = 1.e-2   # Time step for integration
N_bodies = 100  # Number of bodies
N_steps = 10    # Number of simulation steps


# Fix Seed for Initialization
np.random.seed(123)

# Initial Conditions
Masses = np.random.random(N_bodies) * 10
X0 = np.random.random(N_bodies)
Y0 = np.random.random(N_bodies)
PX0 = np.random.random(N_bodies) - 0.5
PY0 = np.random.random(N_bodies) - 0.5

# Initialize
Bodies = [Node(x0, y0, pX0, pY0, masses) for (x0, y0, pX0, pY0, masses) in zip(X0, Y0, PX0, PY0, Masses)]

# Main Model Loop
def Model_Loop_BH(n):
    """
    Run the simulation for a given number of steps.

    Args:
        n (int): Number of steps to run the simulation.
    """
    for i in range(n):
        model_step(Bodies, Theta, G, dt)

# Time the simulation, only works in Google Colab
# %timeit Model_Loop_BH(N_steps)

# Run the Simulation
Model_Loop_BH(N_steps)
