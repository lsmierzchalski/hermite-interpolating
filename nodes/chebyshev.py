from math import cos, pi


def create_chebyshev_nodes(a, b, n):
    nodes = []
    for k in range(n):
        x_k = (a + b) / 2 + (b - a) / 2 * cos((2 * k - 1) / (2 * n) * pi)
        # TODO remove conditional statement
        if k == 0:
            x_k = x_k * -1
        nodes.append(x_k)
    nodes.sort()
    return nodes
