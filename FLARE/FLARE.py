import numpy as np
import warnings
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib
import os


warnings.filterwarnings("ignore")

import sys
sys.path.insert(1, 'src')
sys.path.insert(2, 'src/samplers')

from active_learning_lab import activeLearnerLab
import check_accuracy as ca

batch_size = 10
sampler = 'model_uncertainty'

# X = np.loadtxt('sampled_data/laser_parameters.txt')
# y = np.loadtxt('sampled_data/emissivity_curves.txt')

X = np.load('sampled_data/input_test_data.npy')[:10]
y = np.load('sampled_data/output_test_data.npy')[:10]

max_size = 200
if len(X) >= 200:
    print ('Number of laser parameters reached the max size!')
    sys.exit()

lb = np.array([0.2, 10, 15])
ub = np.array([1.3, 700, 28])

al_setup = activeLearnerLab(X, y,
    lb, ub,
    batch_size,
    sampler,
    verbose=0
)

X_new, model = al_setup.run()

dir_path = 'model'
file_path = os.path.join(dir_path, 'model.pkl')
os.makedirs(dir_path, exist_ok=True)
joblib.dump(model, file_path)




