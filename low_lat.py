import numpy as np
import time

def dot_product_numpy(array_a, array_b):
    return np.dot(array_a, array_b)

a_np = np.array([1, 2, 3])
b_np = np.array([4, 5, 6])
print(f"NumPy Method Result: {dot_product_numpy(a_np, b_np)}")
