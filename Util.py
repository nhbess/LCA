import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

def load_image_as_numpy_array(image_path, normalize=True, black_and_white=False, binary=False, sensibility=0.5):
    image = PIL.Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
    image_array = np.array(image)
    if normalize:
        image_array = image_array / 255.0
    if black_and_white:
        image_array = np.mean(image_array, axis=-1)
    if binary:
        image_array = np.where(image_array > sensibility, 1, 0)
    return image_array

def discretize_target(target:np.array, n_symbols:int) -> np.array:
    target = target/np.max(target)
    symbols = np.arange(n_symbols)
    target = np.abs(target)
    mapped = np.floor(np.abs(target) * len(symbols)).astype(int)
    mapped[mapped == len(symbols)] = len(symbols) - 1  # To handle the edge case when array value is exactly 1
    mapped = (mapped).astype(int)
    return mapped

def load_simple_image_as_numpy_array(image_path):
    image = PIL.Image.open(image_path)
    
    if image.mode == 'RGBA':
        image_array = np.array(image)
        alpha_channel = image_array[:, :, 3]
        image_array = alpha_channel
        image_array = np.where(image_array > 0, 1, 0)
    
    return image_array
