import numpy as np
import os
import sys



laser_parameters = np.load('inconel_data/input_test_data.npy')
emissivity_curves = np.load('inconel_data/output_test_data.npy')
n = 10 #batch size

idx = np.random.choice(np.arange(0, len(laser_parameters), 1), size=n, replace=False) #random selection

parameters_sample = laser_parameters[idx] #sampled parameters
emissivity_sample = emissivity_curves[idx] #sampled emissivity

directory = "sampled_data"

file_path_par = os.path.join(directory, "sampled_parameters.txt")
file_path_em = os.path.join(directory, "sampled_emissivity.txt")

# Check if the directory exists
if not os.path.exists(directory):
    # Create the directory
    os.makedirs(directory)

np.savetxt(file_path_par, parameters_sample) #save the parameters in the sampled_data folder
np.savetxt(file_path_em, emissivity_sample) #save the emissivity in the sampled_data folder