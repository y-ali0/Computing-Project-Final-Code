# Implementation of a quadtree points structure for use in the fast multipole method.

import numpy as np
from itertools import chain

# Define a small epsilon value to handle floating-point arithmetic issues
eps = 7./3 - 4./3 - 1

def _loopchildren(parent):
    """
    Helper function to recursively loop through children nodes.

    Args:
        parent (Node): Parent node.

    Yields:
        Node: Child nodes.
    """
    for child in parent._children:
        if child._children:
            for subchild in _loopchildren(child):
                yield subchild
        yield child


class Node():
    """Single Tree Node"""

    # Constants for neighbour relations:
    # DISREGARD is used to determine whether to skip looking for corner neighbors.
    # If the cardinal neighbors of a cell are larger than that cell and the corresponding
    # neighbor index for the given child index is in DISREGARD, then corner neighbors are not skipped.
    # DISREGARD is a tuple mapping child index to cardinal neighbor index.
    
    DISREGARD = (1, 2, 0, 3)  # Neighbour indices to disregard in certain cases
    
    # CORNER_CHILDREN is a tuple representing child indices corresponding to corner neighbors.
    # When a cardinal neighbor is larger than the cell, CORNER_CHILDREN is consulted
    # to find the child corresponding to the given cardinal neighbor index.
    
    CORNER_CHILDREN = (3, 2, 0, 1)  # Child indices corresponding to corner neighbours

    def __init__(self, width, height, x0, y0, points=None,
                 children=None, parent=None, level=0):
        """
        Initialize a node in the quadtree.

        Args:
            width (float): Width of the node.
            height (float): Height of the node.
            x0 (float): x-coordinate of the bottom-left corner of the node.
            y0 (float): y-coordinate of the bottom-left corner of the node.
            points (list): List of points contained in the node.
            children (list): List of child nodes.
            parent (Node): Parent node.
            level (int): Level of the node in the tree.
        """
        self._points = []  # List to store points contained within the node
        self._children = children  # List to store child nodes
        self._cneighbors = 4 * [None,]  # List to store cardinal neighbours of the node
        self._nneighbors = None  # List to store nearest neighbours of the node
        self._cindex = 0  # Index of the node within its parent's children list
        self.parent = parent  # Parent node
        self.x0, self.y0, self.w, self.h = x0, y0, width, height  # Coordinates and dimensions of the node
        self.verts = ((x0, x0 + width), (y0, y0 + height))  # Vertex coordinates of the node
        self.center = (x0 + width/2, y0 + height/2)  # Center coordinates of the node
        self.level = level  # Level of the node in the tree
        self.inner, self.outer = None, None  # Inner and outer boundaries for interaction calculation

        if points is not None:
            self.add_points(points)

    def __iter__(self):
        """Iterate over child nodes."""
        if self._has_children():
            for child in self._children:
                yield child

    def __len__(self):
        """Return the number of points in the node."""
        if self._points is not None:
            return len(self._points)
        return 0

    def _has_children(self):
        """Check if the node has children."""
        return (self._children is not None)

    def _get_child(self, i):
        """
        Get the i-th child node.

        Args:
            i (int): Index of the child node.

        Returns:
            Node: The i-th child node.
        """
        if self._children is None:
            return self
        return self._children[i]

    def _split(self):
        """Split the node into four children nodes."""
        if self._has_children():
            return

        w = self.w / 2
        h = self.h / 2
        x0, y0 = self.verts[0][0], self.verts[1][0]

        # Create children in order: [NW, NE, SW, SE]
        self._children = [Node(w, h, xi, yi, points=self._points,
                               level=self.level + 1, parent=self)
                          for yi in (y0 + h, y0) for xi in (x0, x0 + w)]
        # Assign child indices for neighbor calculation
        for i, c in enumerate(self._children):
            c._cindex = i

    def _contains(self, x, y):
        """
        Check if a point is contained within the node.

        Args:
            x (float): x-coordinate of the point.
            y (float): y-coordinate of the point.

        Returns:
            bool: True if the point is contained within the node, False otherwise.
        """
        return ((x >= self.verts[0][0] and x < self.verts[0][1]) and
                (y >= self.verts[1][0] and y < self.verts[1][1]))

    def is_leaf(self):
        """Check if the node is a leaf node."""
        return (self._children is None)

    def thresh_split(self, thresh):
        """
        Recursively split the node if it contains more points than the threshold.

        Args:
            thresh (int): Threshold for splitting the node.
        """
        if len(self) > thresh:
            self._split()
        if self._has_children():
            for child in self._children:
                child.thresh_split(thresh)

    def set_cneighbors(self):
        """Set cardinal neighbors for each child node."""
        for i, child in enumerate(self._children):
            # Set sibling neighbors
            sn = (abs(1 + (i^1) - i), abs(1 + (i^2) - i))
            child._cneighbors[sn[0]] = self._children[i^1]
            child._cneighbors[sn[1]] = self._children[i^2]
            # Set other neighbors from parent's neighbors
            pn = tuple(set((0, 1, 2, 3)) - set((sn)))
            nc = lambda j, k: j ^ ((k+1) % 2 + 1)
            child._cneighbors[pn[0]] = (self._cneighbors[pn[0]]._get_child(nc(i, pn[1]))
                                        if self._cneighbors[pn[0]] is not None
                                        else None)
            child._cneighbors[pn[1]] = (self._cneighbors[pn[1]]._get_child(nc(i, pn[0]))
                                        if self._cneighbors[pn[1]] is not None
                                        else None)
            # Recursively set cardinal neighbors
            if child._has_children():
                child.set_cneighbors()

    def add_points(self, points):
        """Add points to the node."""
        if self._has_children():
            for child in self._children:
                child.add_points(points)
        else:
            for d in points:
                if self._contains(d.x, d.y):
                    self._points.append(d)

    def get_points(self):
        """Get all points in the node."""
        return self._points

    def traverse(self):
        """Traverse all nodes in the tree."""
        if self._has_children():
            for child in _loopchildren(self):
                yield child

    @property
    def nearest_neighbors(self):
        """Get nearest neighbors of the node."""
        if self._nneighbors is not None:
            return self._nneighbors

        # Find remaining nearest neighbors of same level
        nn = [cn._cneighbors[(i+1) % 4]
              for i, cn in enumerate(self._cneighbors)
              if cn is not None and cn.level == self.level]
        # Find remaining nearest neighbor at lower levels
        nn += [cn._cneighbors[(i+1) % 4]._get_child(self.CORNER_CHILDREN[i])
               for i, cn in enumerate(self._cneighbors)
               if cn is not None and cn._cneighbors[(i+1) % 4] is not None and
               (cn.level < self.level and i != self.DISREGARD[self._cindex])]

        nn = [n for n in self._cneighbors + nn if n is not None]
        self._nneighbors = nn
        return nn

    def interaction_set(self):
        """
        Get interaction set of the node.

        Returns:
            list: Interaction set of the node.
        """
        nn, pn = self.nearest_neighbors, self.parent.nearest_neighbors
        int_set = []
        for n in pn:
            if n._has_children():
                int_set += [c for c in n if c not in nn]
            elif n not in nn:
                int_set.append(n)
        return int_set


class QuadTree():
    """Quad Tree Class"""

    def __init__(self, points, thresh, bbox=(1, 1), boundary='wall'):
        """
        Initialize a quadtree.

        Args:
            points (list): List of points to build the quadtree from.
            thresh (int): Threshold for splitting nodes.
            bbox (tuple): Bounding box coordinates (width, height).
            boundary (str): Boundary condition for the quadtree ('wall' or 'periodic').
        """
        self.threshold = thresh
        self.root = Node(*bbox, 0, 0)
        if boundary == 'periodic':
            self.root._cneighbors = 4 * [self.root,]
        elif boundary == 'wall':
            self.root._cneighbors = 4 * [None,]
        else:
            raise AttributeError('Boundary of type {} is'
                                 ' not recognized'.format(boundary))
        self._build_tree(points)
        self._depth = None

    def _build_tree(self, points):
        """Build the quadtree."""
        self.root.add_points(points)
        self.root.thresh_split(self.threshold)
        self.root.set_cneighbors()

    def __len__(self):
        """Return the number of points in the quadtree."""
        l = len(self.root)
        for node in self.root.traverse():
            l += len(node)
        return l

    def __iter__(self):
        """Iterate over points in the quadtree."""
        for points in self.root.get_points():
            yield points

    @property
    def depth(self):
        """Get the depth of the quadtree."""
        if self._depth is None:
            self._depth = max([node.level for node in self.root.traverse()])
        return self._depth

    @property
    def nodes(self):
        """Get all nodes in the quadtree."""
        return [node for node in self.root.traverse()]

    def traverse_nodes(self):
        """Traverse all nodes in the quadtree."""
        for node in self.root.traverse():
            yield node


def build_tree(points, tree_thresh=None, bbox=None, boundary='wall'):
    """
    Build a quadtree.

    Args:
        points (list): List of points to build the quadtree from.
        tree_thresh (int): Threshold for splitting nodes.
        bbox (tuple): Bounding box coordinates (width, height).
        boundary (str): Boundary condition for the quadtree ('wall' or 'periodic').

    Returns:
        QuadTree: Quadtree structure.
    """
    if bbox is None:
        coords = np.array([(p.x, p.y) for p in points])
        bbox = (max(coords[:, 0]) + eps, max(coords[:, 1]) + eps)
    if tree_thresh is None:
        tree_thresh = 5

    return QuadTree(points, tree_thresh, bbox=bbox, boundary=boundary)
