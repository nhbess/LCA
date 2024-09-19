import numpy as np

symbols = np.arange(3)
array = np.random.rand(3,3)*2-1
print(symbols)
print(array)

array_sign = np.sign(array)
mapped = np.floor(np.abs(array) * len(symbols)).astype(int)
mapped[mapped == len(symbols)] = len(symbols) - 1  # To handle the edge case when array value is exactly 1
mapped = (mapped * array_sign).astype(int)


print(mapped)