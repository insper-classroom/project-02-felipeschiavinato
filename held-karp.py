import numpy as np
import itertools
from utils import *
from cities import *

def distance(point1, point2):
    return haversine(point1, point2)

def held_karp(cities):
    n = len(cities)

    # Pre-compute distance matrix
    dist = [[distance(cities[i], cities[j]) for j in range(n)] for i in range(n)]

    C = {}
    for k in range(1, n):
        C[(1 << k, k)] = (dist[0][k], [0, k])

    for subset_size in range(2, n):
        print(f"Considering subsets of size {subset_size}")  # Print current subset size
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            

            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dist[m][k], C[(prev, m)][1] + [k]))
                C[(bits, k)] = min(res)
        print(f"Considering subset {subset}")  # Print current subset
        print(f"Minimum distance ending at city {k} is {C[(bits, k)][0]}")  # Print current city and minimum distance

    bits = (2**n - 1) - 1

    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dist[k][0], C[(bits, k)][1] + [0]))
    opt, path = min(res)

    return opt, path

# Use the function:
points = []
for i in range(23):
    points.append((coordinates[i][0], coordinates[i][1]))
print(points)

opt, path = held_karp(points)

print(f"Optimal cost: {opt}")
cities_path = [cities[i] for i in path]
print(f"Path: {cities_path}")
