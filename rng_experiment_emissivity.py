import numpy as np
import os
import sys



emissivity_curves = np.load('inconel_data/output_test_data.npy')
n = 10 #batch size
idx = np.random.choice(np.arange(0, len(emissivity_curves), 1), size=n, replace=False) #random selection
emissivity_sample = emissivity_curves[idx] #sampled emissivity
directory = "sampled_data"
file_path_em = os.path.join(directory, "sampled_emissivity.txt")
if not os.path.exists(directory):
    os.makedirs(directory)
np.savetxt(file_path_em, emissivity_sample) #save the emissivity in the sampled_data folder