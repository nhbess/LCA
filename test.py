import numpy as np

symbols = np.arange(20)
array = np.random.rand(5)
print(symbols)
print(array)

mapped = np.floor(array * len(symbols)).astype(int)
mapped[mapped == len(symbols)] = len(symbols) - 1  # To handle the edge case when array value is exactly 1
print(mapped)