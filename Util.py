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