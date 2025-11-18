import numpy as np
import os
import sys


laser_parameters = np.load('inconel_data/input_test_data.npy')
n = 10 #batch size
idx = np.random.choice(np.arange(0, len(laser_parameters), 1), size=n, replace=False) #random selection
parameters_sample = laser_parameters[idx] #sampled parameters
directory = "sampled_data"
file_path_par = os.path.join(directory, "sampled_parameters.txt")
if not os.path.exists(directory):
    os.makedirs(directory)

np.savetxt(file_path_par, parameters_sample) #save the parameters in the sampled_data folder
