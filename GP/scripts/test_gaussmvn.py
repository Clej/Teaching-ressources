from utils import numpy_to_latex
import numpy as np

variance = 1.5
sigma_high = np.array(
    [[variance, 0.99, 0.98, 0.96, 0.94],
     [0.99, variance, 0.90, 0.98, 0.96],
     [0.98, 0.99, variance, 0.99, 0.98],
     [0.96, 0.98, 0.99, variance, 0.99],
     [0.94, 0.96, 0.98, 0.99, variance]]
)

print(
    numpy_to_latex(sigma_high)
)