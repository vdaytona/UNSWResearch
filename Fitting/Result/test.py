import numpy as np

x = np.asarray(range(10))

print x[:-1]
print x[1:]
print x[2:-1] / x[1:-2]