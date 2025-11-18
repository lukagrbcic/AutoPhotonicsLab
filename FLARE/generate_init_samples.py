import numpy as np
import warnings
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

warnings.filterwarnings("ignore")

import sys
sys.path.insert(1, 'src')
sys.path.insert(2, 'src/samplers')

from active_learning_lab import activeLearnerLab


init_size = 10
batch_size = 10

sampler = 'lhs'

X = []
y = []

lb = np.array([0.2, 10, 15])
ub = np.array([1.3, 700, 28])

test_data = []

al_setup = activeLearnerLab(X, y,
    lb, ub, batch_size,
    sampler, test_data, init_size,
    verbose=0
)

X_init = al_setup.initialize()

# dir_path = 'sampled_data'
# file_path = os.path.join(dir_path, 'laser_parameters.txt')
# os.makedirs(dir_path, exist_ok=True)

# np.savetxt(file_path, X_init)






