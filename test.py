import torch

# Input array and multiplier
array = torch.Tensor([0, -1, 1])
M = 1  # Multiplier for scaling

# Differentiable mappings
negatives = torch.relu(-array) # Softmax function for positive values
positives = torch.relu(array) # Softmax function for positive values
zeros = torch.exp(-array*array)               # Gaussian-like function for zeros
signs = torch.tanh(array * M)               # Smooth sign function

# Print intermediate results
print("Negatives:", negatives)
print("Positives:", positives)
print("Zeros:", zeros)
print("Signs:", signs)

# Combining the filters using a sum of products
mapped_products = torch.sum(torch.stack([negatives, -zeros, positives]), dim=0) * signs
print("Mapped Products:", mapped_products)
