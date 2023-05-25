import random
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from cities import *

random.seed(1)

nodes = [i for i in range(23)]

distances = []

for _ in range(1000000):
    random.shuffle(nodes)
    total_distance = sum(haversine(coordinates[nodes[i]], coordinates[nodes[i+1]]) for i in range(22))
    total_distance += haversine(coordinates[nodes[22]], coordinates[nodes[0]])
    distances.append(total_distance)

plt.hist(distances, bins=100)
plt.show()
