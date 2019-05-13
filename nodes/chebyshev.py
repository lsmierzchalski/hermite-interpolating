from math import cos, pi

from numpy import array


def create_chebyshev_nodes(a, b, n):
    nodes = []
    for k in range(1, n+1):
        x_k = (a + b) / 2 + (b - a) / 2 * cos((2 * k - 1) / (2 * n) * pi)
        nodes.append(x_k)
    nodes.sort()
    return array(nodes)

